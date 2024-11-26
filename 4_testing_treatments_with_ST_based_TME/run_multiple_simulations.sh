MAX_CPU_USAGE=90 # maximum CPU usage
CHECK_INTERVAL=60 # check interval = 60s
# all ST data names {241_C,242,242_C,243,248,248_C,251,255,256_C,259,259_C,260,262,265,266,268,269,270,275,296,304,313,334,334_C}
for ST_data_folder in {241_C,262,270} # selected ST data-based TME 
do
  for set in {0..99} # for each individual (N = 100)
  do  
    while true
    do
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}') # current CPU usage
        current_jobs=$(ps aux | grep -E "treatment_ST" | grep -v grep | wc -l) # current number of 'treatment_ST' jobs
    
        echo "current CPU usage: ${cpu_usage}% current number of jobs: $current_jobs"
        if (( $(echo "$cpu_usage < $MAX_CPU_USAGE" | bc -l) )); then
            break  
        fi

        sleep $CHECK_INTERVAL
    done     
    echo "current CPU usage: ${cpu_usage}% current number of jobs: $current_jobs"
    
    # start treatment_ST jobs 
    nohup ./bin_ST/optimalRL_treatment_ST  optimalRL_treatment_ST_output  $set 1 200 1 100 $ST_data_folder &
    nohup ./bin_ST/CSF1RIonly_treatment_ST  CSF1RIonly_treatment_ST_output  $set 1 200 1 100 $ST_data_folder &
    nohup ./bin_ST/IGF1RIonly_treatment_ST  IGF1RIonly_treatment_ST_output  $set 1 200 1 100 $ST_data_folder &
    nohup ./bin_ST/CSF1RIandIGF1RI_treatment_ST  CSF1RIandIGF1RI_treatment_ST_output  $set 1 200 1 100 $ST_data_folder &

    # ./bin_ST/optimalRL_treatment_ST \    # path to the executable file for optimal RL treatment with ST-based data
    # optimalRL_treatment_ST_output \      # output folder name to store the population results (argv[1])
    # $set \                               # output set name to store each individual results (argv[2])
    # 1 \                                  # number of simulations for an individual (argv[3])
    # 200 \                                # simulation time 200 days (argv[4])
    # 1 \                                  # whether to enable randomness for the key parameters, 1 for enable and 0 for disable (argv[5])
    # 100 \                                # cell spot size 100 * 100 (argv[6])
    # $ST_data_folder                      # ST data-based TME save folder (argv[7])
  done
done
