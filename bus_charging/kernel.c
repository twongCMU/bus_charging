#include <curand_kernel.h>
#include <stdlib.h>


__constant__ unsigned int routes_lengths[NUM_ROUTES];
__constant__ unsigned int stops_lengths[NUM_STOPS];

extern "C" {
  __device__ void move_charger(unsigned int rounds,
			       curandState_t *s,
			       unsigned int *stops_with_chargers,
			       unsigned int *stop_data,
			       unsigned int *stop_id_to_move,
			       unsigned int *new_charger_stop_id)
  {
    /*
      We want to move a random charger.
      
      First we'll randomly pick the i-th charger in stops_with_chargers to get the stop_id
      then we'll randomly pick a route that has that stop and move the charger in a random direction from that
    */
    *stop_id_to_move = NUM_STOPS;
    unsigned int charger_offset = curand(s) % NUM_CHARGERS;
    unsigned int charger_offset_start = charger_offset;
    for (int i = 0; i < NUM_STOPS; i++) {
      if ((stops_with_chargers[(int)i/32] >> ((int)i%32)) & 0x1 == 1U) {
	if (charger_offset == 0){
	  *stop_id_to_move = i;
	  break;
	}
	charger_offset--;
      }
    }
    if (*stop_id_to_move == NUM_STOPS) {
      printf("ERROR: in selecting random charger, failed to find a stop_id_to_move offset is %u started as %u round %u\n", charger_offset, charger_offset_start, rounds);
    }
    
    unsigned int route_offset = curand(s) % stops_lengths[*stop_id_to_move];
    // in the stop_data, the uint packs the route_id in the upper 16 bits and the stop index into the lower 16
    unsigned int route_id_to_move = stop_data[(*stop_id_to_move*LONGEST_STOPS) + route_offset];
    
    //int route_id_to_move_index = route_id_to_move & ((1<<16)-1);
    route_id_to_move = (route_id_to_move >> 16) & ((1<<16)-1);
    //printf("route_id_to_move %u for stop_id %u route index %u\n", route_id_to_move, *stop_id_to_move, route_id_to_move_index);
    // move it in this direction - 0 = forward, 1 = backward
    int charger_move_direction = ((curand(s) % 2) == 0 ? 1 : -1);
    
    // pick a new location for one charger and keep picking until we have a location without a charger
    int keep_moving  = 1;
    *new_charger_stop_id = 0;
    while (keep_moving > 0) {
      *new_charger_stop_id = curand(s) % NUM_STOPS;
      keep_moving = ((stops_with_chargers[(int)*new_charger_stop_id/32] >> ((int)*new_charger_stop_id%32)) & 0x1);
      //printf("Keep moving %u for stop id %u thread %u\n", keep_moving, *new_charger_stop_id, threadIdx.x);
    }

    
  }


  __device__ float calculate_utility(struct route_stop_s *route_data,
				     unsigned int *stops_with_chargers,
				     unsigned int stop_id_to_move,
				     unsigned int new_charger_stop_id)
  {
    // now we calculate the utility for the routes given this change
    // we do this by assuming the bus has enough charge to do exactly one loop on its route
    // then we drive it one loop to see how much battery it has left compared to when it started
    // if it has more, then we will never run out so utility is 1.0
    // if it is less, that means we used some % of the battery we had brought with us. The % we have left
    // is the utility, so if we have 90% charge left, our utility is 0.90. If we have 10% of our battery
    // left, that's low utility, so 0.10
    // the total utility for all routes is the minimum utility of all of the routes
    // if we took the average, some good routes would overshadow a bad one
    // this is especially true since utility grows faster with chargers for shorter routes due to the way we calculate it
    float total_utility = 1.0f;
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
	  
	  //if (threadIdx.x==0 && blockIdx.x==0) {
	    //printf("saw charger at stop id %u\n", this_station_id);
	    //}
	  chargers_seen++;
	}
      }

      // we assume the bus started its route with exactly enough power to finish the route
      charge_actual += charge_required;
      if (charge_actual < 0.0) { // I think this is actually impossible. How can a bus use more energy than a loop required. Keep this for floating point weirdness
	if (total_utility > 0.0) {
	  total_utility = 0.0;
	}
      }
      else if (charge_actual >= charge_required) {
	// if we didn't lose charge after driving around, that's maximum utility because we'll never run out
	if (total_utility > 1.0) {
	  total_utility = 1.0;
	}
      }
      else {
	// if we're here, charge_actual is >=0 and < charge_required
	// the utility for this configuration is the % of charge we have left after doing a loop
	if (total_utility > charge_actual/charge_required) {
	  total_utility = charge_actual/charge_required;
	}
      }
      /*
      if (threadIdx.x==0 && blockIdx.x==0) {
	printf("Utility after this round was %f %f %f  thread %u run %u for route %u saw %u chargers\n", total_utility, charge_actual, charge_required, threadIdx.x, blockIdx.x, route_id, chargers_seen);
	}*/
    }

    return total_utility;
  }
  
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

    // copy the initial charger arrangement into shared memory for performance
    if (threadIdx.x == 0) {
      for (int j = 0; j < NUM_STOPS_INTS; j++) {
	stops_with_chargers[j] = stops_with_chargers_ret[blockIdx.x * NUM_STOPS_INTS + j];
      }
    }
	
    float previous_utility = 0;
    for (int rounds = 0; rounds < ROUNDS; rounds++) {
      // assume 1 permutation per thread to keep code simple
      unsigned int stop_id_to_move;
      unsigned int new_charger_stop_id;

      move_charger(rounds, &s, stops_with_chargers, stop_data, &stop_id_to_move, &new_charger_stop_id);
      // At this point we know we're moving a charger from stop_id_to_move to new_charger_stop_id
      
      // store the utility in shared memory so we can pick the best one
      permutation_utilities[threadIdx.x] = calculate_utility(route_data, stops_with_chargers, stop_id_to_move, new_charger_stop_id);

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
      if (best_utility > previous_utility || (best_utility/previous_utility) > annealing_threshold) {
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

  __device__ void increment_charger_list(unsigned int *charger_permutation_counter,
					 unsigned int *stops_with_chargers){

    int i;
    for (i = NUM_CHARGERS - 1; i >= 0; i--) {
      // clear the old charger at stops_with_chargers[i]
      stops_with_chargers[(int)charger_permutation_counter[i]/32] &= (~(0x1<<((int)charger_permutation_counter[i]%32)));

      // move the charger at stops_with_chargers[i] to the next one
      charger_permutation_counter[i]++;
      // if this is legal, then we're done. If not continue the loop to the next charger
      if (charger_permutation_counter[i] < NUM_STOPS - (NUM_CHARGERS -1 - i)) {
	break;
      }
    }

    stops_with_chargers[(int)charger_permutation_counter[i]/32] |= (0x1<<((int)charger_permutation_counter[i]%32));
    // then walk back to the right of the charger array and update them
    for (i = i+1; i < NUM_CHARGERS; i++) {
      charger_permutation_counter[i] = charger_permutation_counter[i-1]+1;
      stops_with_chargers[(int)charger_permutation_counter[i]/32] |= (0x1<<((int)charger_permutation_counter[i]%32));
    }
  }


  __device__ void increment_charger_list_initial(unsigned int *charger_permutation_counter,
					 unsigned int *stops_with_chargers){

    /*
    if (threadIdx.x == 29) {
      printf("incrementing (num stops %u num chargers %u:", NUM_STOPS, NUM_CHARGERS);
      for (int i = 0 ;i <NUM_CHARGERS; i++) {
	printf("[%u]",charger_permutation_counter[i]);
      }
      printf("\n");
    }
    */
    int i;
    // find the leftmost charger in the array that needs to be incremented
    for (i = NUM_CHARGERS - 1; i >= 0; i--) {
      // move the charger at stops_with_chargers[i] to the next one
      charger_permutation_counter[i]++;

      // if this is legal, then we're done
      if (charger_permutation_counter[i] < NUM_STOPS - (NUM_CHARGERS - 1 - i)) {
	break;
      }
    }

    // then walk back to the right of the charger array and update them
    for (i = i+1; i < NUM_CHARGERS; i++) {
      charger_permutation_counter[i] = charger_permutation_counter[i-1]+1;
    }
    
    /*
    if (threadIdx.x == 29) {
      printf("now :");
      for (int i = 0; i <NUM_CHARGERS; i++) {
	printf("[%u]",charger_permutation_counter[i]);
      }
      printf("\n");
    }
    */
  }

  __global__ void run_brute_force(struct route_stop_s *route_data,
				    unsigned int *stop_data,
				    float* final_utility_ret,
				    unsigned int *stops_with_chargers_ret) {
    __shared__ float utilities[32];
    
    unsigned int stops_with_chargers[NUM_STOPS_INTS];
    unsigned int stops_with_chargers_best[NUM_STOPS_INTS];
    unsigned int charger_permutation_counter[NUM_CHARGERS];

    for (int j = 0; j < NUM_STOPS_INTS; j++) {
      stops_with_chargers[j] = 0;
    }

    for (int j = 0; j < NUM_CHARGERS; j++) {
      charger_permutation_counter[j] = j;
    }

    
    unsigned long long start_offset = NUM_WORK_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
    if (start_offset + NUM_WORK_PER_THREAD > TOTAL_WORK) {
      printf("INFO: Thread %u block %u start offset would exceed total work. Resetting\n", threadIdx.x, blockIdx.x);
      start_offset = LAST_THREAD_START_OFFSET;
    }
    //printf("thread %u, start offset %u\n", threadIdx.x, start_offset);
    float best_utility = 0.0f;

    // set charger_permutation_counter to the first offset we will check
    for (unsigned long long i = 0; i < start_offset; i++) {
      increment_charger_list_initial(charger_permutation_counter, stops_with_chargers);
    }
    
    for (int i = 0; i < NUM_CHARGERS; i++) {
      if (charger_permutation_counter[i] > NUM_STOPS) {
	printf("ERROR, got charger ID %u with max %u for thread %u at i %u\n", charger_permutation_counter[i], NUM_STOPS, threadIdx.x, i);
      }
      stops_with_chargers[(int)charger_permutation_counter[i]/32] |= (0x1<<((int)charger_permutation_counter[i]%32));
    }

    for (int i = 0; i < NUM_WORK_PER_THREAD; i++) {
      // calculate_utility assumes we have a charging station move pending
      // but here we don't, so we fake one by saying we moved a station from
      // a station back to itself
      float utility = calculate_utility(route_data,
					stops_with_chargers,
					NUM_STOPS+1,
					NUM_STOPS+2);
      
      if (utility > best_utility) {
	best_utility = utility;
       
	for (int j = 0; j < NUM_STOPS_INTS; j++) {
	  stops_with_chargers_best[j] = stops_with_chargers[j];
	}	
      }

      // on the last work item, don't bother incrementing since we won't check the result
      // even worse, we'll stomp memory as the last thread tries to roll over and gets confused
      if (i < NUM_WORK_PER_THREAD-1) {
	increment_charger_list(charger_permutation_counter, stops_with_chargers);
      }
    }
    
    utilities[threadIdx.x] = best_utility;
    int best_index = 0;
    best_utility = utilities[0];
    
    for (int i = 0; i < 32; i++) {
      /*
      if (blockIdx.x==1 && threadIdx.x==0) {
	printf("Looking at idx %u utility %f, best so far %f at index %u\n", i, utilities[i], best_utility, best_index);
      }
      */
      if (utilities[i] > best_utility) {

	best_utility = utilities[i];
	best_index = i;

	//if (blockIdx.x==1 && threadIdx.x==0) {
	// printf("at index %u doing update\n", i);
	//}
	
      }
    }
    if (threadIdx.x == best_index) {
      //printf("Winner is idx %u block %u utility %f\n", best_index, blockIdx.x, best_utility);
      final_utility_ret[blockIdx.x] = best_utility;
      for (int j = 0; j < NUM_STOPS_INTS; j++) {
	stops_with_chargers_ret[blockIdx.x * NUM_STOPS_INTS + j] = stops_with_chargers_best[j];
      }
    }
    
  }
}





	  

      
