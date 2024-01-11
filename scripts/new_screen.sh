screen_name=${1}

screen -S ${screen_name} -L -Logfile screen_log/${screen_name}.log
