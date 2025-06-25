# Некоторые инструменты

## 1. Профилировщик pytorch
Оценка слабых мест модели  
В `profiler.py` необходимо передать свою модель в метод `profile`, затем запустить.  

Полученные логи можно открыть в `TensorBoard` или в `chrome://tracing/`

## 2. Разблокировать Performance Mode RKNN
Запуск на RKNN устройстве
```bash
sh rknn_unlock_performance.sh
```
Пояснения:  
Выставим максимальные частоты для CPU и NPU:
1. Установка CPU governor в performance
```bash
# Для всех CPU ядер
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    echo performance | sudo tee $cpu/cpufreq/scaling_governor
done
```
2. Установка NPU governor (имя устройства может отличаться)
```bash
# Найдём имя NPU устройства:
ls /sys/class/devfreq/

# Например:
echo performance | sudo tee /sys/class/devfreq/fd8c0000.npu/governor
```

3. Убедимся, что частота максимальная
```
cat /sys/class/devfreq/fd8c0000.npu/cur_freq
cat /sys/class/devfreq/fd8c0000.npu/available_frequencies
```