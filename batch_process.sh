# !/bin/bash

for i in {0..511..1}
do
    if [ $i -le 63 ]
    then
        python center_lucent_optimize.py --network resnet18 --basedir /media/andrelongon/DATA/feature_viz/new_test  --module layer1.0 --neuron $i --jitter 0
        python center_lucent_optimize.py --network resnet18 --basedir /media/andrelongon/DATA/feature_viz/new_test --module layer1.1.bn2 --neuron $i --jitter 0
        python center_lucent_optimize.py --network resnet18 --basedir /media/andrelongon/DATA/feature_viz/new_test --module layer1.1 --neuron $i --jitter 0
    fi

    if [ $i -le 127 ]
    then
        python center_lucent_optimize.py --network resnet18  --module layer2.0.downsample.1 --neuron $i --jitter 0
        python center_lucent_optimize.py --network resnet18  --module layer2.0.bn2 --neuron $i --jitter 0
        python center_lucent_optimize.py --network resnet18  --module layer2.0 --neuron $i --jitter 0

        python center_lucent_optimize.py --network resnet18 --module layer2.1.bn2 --neuron $i --jitter 4
        python center_lucent_optimize.py --network resnet18 --module layer2.1 --neuron $i --jitter 4
    fi

    if [ $i -le 255 ]
    then
        python center_lucent_optimize.py --network resnet18  --module layer3.0.downsample.1 --neuron $i --jitter 4
        python center_lucent_optimize.py --network resnet18  --module layer3.0.bn2 --neuron $i --jitter 4
        python center_lucent_optimize.py --network resnet18  --module layer3.0 --neuron $i --jitter 4

        python center_lucent_optimize.py --network resnet18 --module layer3.1.bn2 --neuron $i
        python center_lucent_optimize.py --network resnet18 --module layer3.1 --neuron $i
    fi

    python center_lucent_optimize.py --network resnet18  --module layer4.0.downsample.1 --neuron $i
    python center_lucent_optimize.py --network resnet18  --module layer4.0.bn2 --neuron $i
    python center_lucent_optimize.py --network resnet18  --module layer4.0 --neuron $i

    python center_lucent_optimize.py --network resnet18 --module layer4.1.bn2 --neuron $i
    python center_lucent_optimize.py --network resnet18 --module layer4.1 --neuron $i
done

python tuning_curve.py --network resnet18 --layer layer1.0
python tuning_curve.py --network resnet18 --layer layer1.1.bn2
python tuning_curve.py --network resnet18 --layer layer1.1 
python tuning_curve.py --network resnet18 --layer layer2.0.downsample.1
python tuning_curve.py --network resnet18 --layer layer2.0.bn2
python tuning_curve.py --network resnet18 --layer layer2.0
python tuning_curve.py --network resnet18 --layer layer2.1.bn2
python tuning_curve.py --network resnet18 --layer layer2.1
python tuning_curve.py --network resnet18 --layer layer3.0.downsample.1
python tuning_curve.py --network resnet18 --layer layer3.0.bn2
python tuning_curve.py --network resnet18 --layer layer3.0
python tuning_curve.py --network resnet18 --layer layer3.1.bn2
python tuning_curve.py --network resnet18 --layer layer3.1
python tuning_curve.py --network resnet18 --layer layer4.0.downsample.1
python tuning_curve.py --network resnet18 --layer layer4.0.bn2
python tuning_curve.py --network resnet18 --layer layer4.0
python tuning_curve.py --network resnet18 --layer layer4.1.bn2
python tuning_curve.py --network resnet18 --layer layer4.1