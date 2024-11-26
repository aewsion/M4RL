MAX_CPU_USAGE=90 # maximum CPU usage
CHECK_INTERVAL=60 # check interval = 60s

for set in {0..99} # for each individual
do  
  while true
  do
      cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}') # current CPU usage
      current_jobs=$(ps aux | grep -E "continuous_CSF1R_I_treatment_case" | grep -v grep | wc -l) # current number of 'continuous_CSF1R_I_treatment_case' jobs
  
      echo "current CPU usage: ${cpu_usage}% current number of jobs: $current_jobs"
      if (( $(echo "$cpu_usage < $MAX_CPU_USAGE" | bc -l) )); then
          break  
      fi

      sleep $CHECK_INTERVAL
  done     
  echo "current CPU usage: ${cpu_usage}% current number of jobs: $current_jobs"
  
  # start continuous_CSF1R_I_treatment_case jobs 
  nohup ./bin/continuous_CSF1R_I_treatment_case continuous_CSF1R_I_treatment_case_output $set 1 200 1 &

  # ./bin/continuous_CSF1R_I_treatment_case \       # path to the executable file for continuous CSF1R_I treatment case
  # continuous_CSF1R_I_treatment_case_output \      # output folder name to store the population results (argv[1])
  # $set \                                          # output set name to store each individual results (argv[2])
  # 1 \                                             # number of simulations for an individual (argv[3])
  # 200 \                                           # simulation time 200 days (argv[4])
  # 1                                               # whether to enable randomness for the key parameters, 1 for enable and 0 for disable (argv[5])
done

