# Study to make a General installer

The idea is to make a script that will run the scripts in order and additionally gives options to what to install.

1. setup-novnc.sh 
2. setup-conda.sh 
3. setup-isaacsim.sh 
4. setup-isaaclab.sh 
5. setup-gr00t.sh (optional)
6. setup-leisaac.sh (optional)

Need to study how to install IsaacSim Extensions and IsaacLab Projects.


Need to add a script to perform the post install and the cache warmup with IsaacSim.

```bash
post_install.sh

warmup.sh
```


IsaacSim Flags

* --ext-folder PATH : Add extension folder to look extensions in.
* --ext-path PATH : Add direct extension path (allows adding single extension).


## Making installer run after reboot

The following is an example using custom functions need to improve.

```bash
cat ~/status-reboot.sh
```

```bash
#!/bin/bash

STATUS_LOG="$HOME/our.status.log"

# Determinate whether the log file exists ? get the status : set status0
if [[ -f $STATUS_LOG ]]
then
        CURRENT_STATUS="$(cat "$STATUS_LOG")"
else
        CURRENT_STATUS="stage0"
        echo "$CURRENT_STATUS : $(date)"
        echo "$CURRENT_STATUS" > "$STATUS_LOG"
        # You could reboot at this point,
        # but probably you want to do action_1 first
fi

# Define your actions as functions
action_1()
{
        # do the 1st action

        CURRENT_STATUS="stage1"
        echo "$CURRENT_STATUS : $(date)"
        echo "$CURRENT_STATUS" > "$STATUS_LOG"
        exit # You could reboot at this point
}

action_2()
{
        # do the 2nd action

        CURRENT_STATUS="stage2"
        echo "$CURRENT_STATUS : $(date)"
        echo "$CURRENT_STATUS" > "$STATUS_LOG"
        exit # You could reboot at this point
}

case "$CURRENT_STATUS" in
stage0)
  action_1
  ;;
stage1)
  action_2
  ;;
stage2)
  echo "The script '$0' is finished."
  ;;
*)
  echo "Something went wrong!"
  ;;
esac
```