#include <curand_kernel.h>
#include <stdlib.h>


__constant__ unsigned int routes_lengths[NUM_ROUTES];
__constant__ unsigned int stops_lengths[NUM_STOPS];

extern "C" {
  __global__ void run_approximation(struct route_stop_s *route_data,
				    unsigned int *stop_data,
				    curandState_t* rand_states,
				    float* final_utility_ret,
				    unsigned int *stops_with_chargers_ret) {

    __shared__ unsigned int stops_with_chargers[NUM_STOPS_INTS];
    __shared__ float stops_with_chargers_utility;
    __shared__ float permutation_utilities[32];
    
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    float annealing_threshold = SA_THRESHOLD_INITIAL;
    
    // this costs 7-10 registers, but it is probably worth it to not have to hit main memory
    curandState_t s = rand_states[tidx];

    // copy the initial charger arrangement
    if (threadIdx.x == 0) {
      for (int j = 0; j < NUM_STOPS_INTS; j++) {
	stops_with_chargers[j] = stops_with_chargers_ret[blockIdx.x * NUM_STOPS_INTS + j];
      }
    }
	
    float previous_utility = 0;
    for (int rounds = 0; rounds < ROUNDS; rounds++) {
      // assume 1 permutation per thread to keep code simple


      /*
	We want to move a random charger.

	First we'll randomly pick the i-th charger in stops_with_chargers to get the stop_id
	then we'll randomly pick a route that has that stop and move the charger in a random direction from that
      */
      unsigned int stop_id_to_move = NUM_STOPS;
      unsigned int charger_offset = curand(&s) % NUM_CHARGERS;
      unsigned int charger_offset_start = charger_offset;
      for (int i = 0; i < NUM_STOPS; i++) {
	if ((stops_with_chargers[(int)i/32] >> ((int)i%32)) & 0x1 == 1U) {
	  if (charger_offset == 0){
	    stop_id_to_move = i;
	    break;
	  }
	  charger_offset--;
	}
      }
      if (stop_id_to_move == NUM_STOPS) {
	printf("ERROR: in selecting random charger, failed to find a stop_id_to_move offset is %u started as %u round %u\n", charger_offset, charger_offset_start, rounds);
      }

      unsigned int route_offset = curand(&s) % stops_lengths[stop_id_to_move];
      // in the stop_data, the uint packs the route_id in the upper 16 bits and the stop index into the lower 16
      unsigned int route_id_to_move = stop_data[(stop_id_to_move*LONGEST_STOPS) + route_offset];
      
      //int route_id_to_move_index = route_id_to_move & ((1<<16)-1);
      route_id_to_move = (route_id_to_move >> 16) & ((1<<16)-1);
      //printf("route_id_to_move %u for stop_id %u route index %u\n", route_id_to_move, stop_id_to_move, route_id_to_move_index);
      // move it in this direction - 0 = forward, 1 = backward
      int charger_move_direction = ((curand(&s) % 2) == 0 ? 1 : -1);

      /*
      int keep_moving = 1;
      unsigned int new_charger_stop_id = 0;
      while (keep_moving > 0) {
	route_id_to_move_index += charger_move_direction;
	// if we are at the end of the route list, wrap around to the first stop on the route
	if (route_id_to_move_index == routes_lengths[route_id_to_move]) {
	  route_id_to_move_index = 0;
	}
	// if we went past the beginning of the route list, wrap around to the last stop on the route
	if (route_id_to_move_index < 0) {
	  route_id_to_move_index = routes_lengths[route_id_to_move]-1;
	}
	// if we want to move the charger to a stop that already has a charger, move it again
	// it is kind of expensive to do this in GPU but it should be rare, and I'd rather get a permutation
	// that's useful than do nothig
	new_charger_stop_id = route_data[(route_id_to_move* LONGEST_ROUTE) +route_id_to_move_index].station_id;
	keep_moving = ((stops_with_chargers[(int)new_charger_stop_id/32] >> ((int)new_charger_stop_id%32)) & 0x1);
      }
      */
      
      int keep_moving  = 1;
      unsigned int new_charger_stop_id = 0;
      while (keep_moving > 0) {
	new_charger_stop_id = curand(&s) % NUM_STOPS;
	keep_moving = ((stops_with_chargers[(int)new_charger_stop_id/32] >> ((int)new_charger_stop_id%32)) & 0x1);
	//printf("Keep moving %u for stop id %u thread %u\n", keep_moving, new_charger_stop_id, threadIdx.x);
      }
      // At this point we know we're moving a charger from stop_id_to_move to new_charger_stop_id


      // now we calculate the utility for the routes given this change
      // we do this by assuming the bus has enough charge to do exactly one loop on its route
      // then we drive it one loop to see how much battery it has left compared to when it started
      // if it has more, then we will never run out so utility is 1.0
      // if it is less, that means we used some % of the battery we had brought with us. The % we have left
      // is the utility, so if we have 90% charge left, our utility is 0.90. If we have 10% of our battery
      // left, that's low utility, so 0.10
      float total_utility = 0.0f;
      for (int route_id = 0; route_id < NUM_ROUTES; route_id++) {
	float charge_actual = 0.0f;
	float charge_required = 0.0f;
	unsigned int chargers_seen = 0;
	for (int stop_offset = 0; stop_offset < routes_lengths[route_id]; stop_offset++) {
	  // deduct the energy we use driving to the next stop
	  charge_actual -= ENERGY_USED_PER_KM * route_data[(route_id *LONGEST_ROUTE) +stop_offset].distance_m/1000.0;
	  charge_required += ENERGY_USED_PER_KM * route_data[(route_id *LONGEST_ROUTE) +stop_offset].distance_m/1000.0;
	  
	  unsigned int this_station_id = route_data[(route_id* LONGEST_ROUTE) +stop_offset].station_id;

	  // if we moved the station here we can charge OR
	  // if we didn't move the station away from this location and there is a charger here, also charge	  
	  if (this_station_id == new_charger_stop_id ||
	      (this_station_id != stop_id_to_move && (stops_with_chargers[(int)this_station_id/32] >> ((int)this_station_id%32)) & 0x1 == 1U)) { 
	    charge_actual += ENERGY_CHARGE_PER_MINUTE * STOP_WAIT_MINUTES;
	    chargers_seen++;
	  }
	}

	// we assume the bus started its route with exactly enough power to finish the route
	charge_actual += charge_required;
	if (charge_actual < 0.0) { // I think this is actually impossible. How can a bus use more energy than a loop required. Keep this for floating point weirdness
	  continue; // add 0 to total_utility
	}
	else if (charge_actual >= charge_required) {
	  // if we didn't lose charge after driving around, that's maximum utility because we'll never run out
	  total_utility += 1.0;
	}
	else {
	  // if we're here, charge_actual is >=0 and < charge_required
	  // the utility for this configuration is the % of charge we have left after doing a loop
	  total_utility += charge_actual/charge_required;
	}
	//printf("Utility after this round was %f %f %f round %u thread %u run %u for route %u saw %u chargers\n", total_utility, charge_actual, charge_required, rounds, threadIdx.x, blockIdx.x, route_id, chargers_seen);
      }

      // the total utility is the average of all the route utilities
      total_utility /= NUM_ROUTES;


      // store the utility in shared memory so we can pick the best one
      permutation_utilities[threadIdx.x] = total_utility;

      // find which thread had the best utility
      // if there is a tie the first one wins but it doesn't matter because
      // the permutations are generated randomly already
      float best_utility = 0.0f;
      int best_index = -1;
      for (int i = 0; i < 32; i++) {
	if (permutation_utilities[i] > best_utility) {
	  best_utility = permutation_utilities[i];
	  best_index = i;
	}
      }

      // if this utility is higher than before or it is better than the threshold, keep it
      if (best_utility > previous_utility || best_utility > annealing_threshold) {
	if (threadIdx.x == best_index) {
	  // the winning thread updates stops_with_chargers by removing the bit from stop_id_to_move and setting it on new_charger_stop_id
	  //printf("moving from stop %u to %u (index %u) along route %u for round %u\n", stop_id_to_move, new_charger_stop_id, route_id_to_move_index,route_id_to_move, rounds);
	  stops_with_chargers[(int)stop_id_to_move/32] &= (~(0x1<<((int)stop_id_to_move%32)));
	  stops_with_chargers[(int)new_charger_stop_id/32] |= (0x1<<((int)new_charger_stop_id%32));
	  stops_with_chargers_utility = best_utility;
	}
	previous_utility = best_utility;
      }

      // update annealing info
      if (((rounds+1) % ANNEALING_COOLING_FREQUENCY) == 0) {
	annealing_threshold += ANNEALING_COOLING_STEP;
      }
      if (annealing_threshold > 1.0) {
	annealing_threshold = 1.0;
	// the model should calculate the cooling step in a way that this never happens
	if (ROUNDS-rounds > 50) {
	  printf("ERROR: annealing threshold unexpectedly over 1.0 at round %u\n", rounds);
	}
      }
    } // done with rounds

    // find which permutation had the best final utility and save it
    if (threadIdx.x == 0) {
      final_utility_ret[blockIdx.x] = stops_with_chargers_utility;
      
      // copy the best charger arrangement
      for (int j = 0; j < NUM_STOPS_INTS; j++) {
	stops_with_chargers_ret[blockIdx.x * NUM_STOPS_INTS + j] = stops_with_chargers[j];
      }
    }

    rand_states[tidx] = s;
  }
}

	



	  

      
