#! /bin/bash

if [ "fin" == "$1" ]; then
  echo "("
  echo "(benchmark-date end \"`date`\")"
  echo ")"
  exit 0
fi

if test -n "$USER"; then 
   user="$USER"
elif test -n "$LOGNAME"; then
   user="$LOGNAME";
else
   user=`whoami`
fi;

lav=`(uptime || w)  2>/dev/null | sed 1q`
hostname=`(hostname || uname -n) 2>/dev/null | sed 1q`

echo "("
echo "(benchmark-date start \"`date`\")"
echo "(benchmark-user \"${user}\")"
echo "(load-average \"${lav}\")"
echo "(hostname \"${hostname}\")"
echo ")"

nvidia-smi
