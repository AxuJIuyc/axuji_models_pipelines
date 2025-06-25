#!/bin/bash
echo "[⚡] Setting performance mode..."
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    echo performance > $cpu/cpufreq/scaling_governor
done

echo performance > /sys/class/devfreq/fd8c0000.npu/governor
