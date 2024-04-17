IO="io_parallel"
ST="Latency"
#for i in 128; do
#    for o in 128; do
#        for r in $(seq 1 $i); do
#            IN=$i OU=$o RF=$r make run-ml-hls
#        done
#    done
#done
#for i in 32; do
#    for o in 32; do
#        for r in $(seq 1 $i); do
#            IN=$i OU=$o RF=$r make run-ml-hls
#        done
#    done
#done
#for i in 16; do
#    for o in 16; do
#        for r in $(seq 1 $i); do
#            IN=$i OU=$o RF=$r make run-ml-hls
#        done
#    done
#done
for i in 4; do
    for o in 4; do
        for r in $(seq 1 $i); do
            IN=$i OU=$o RF=$r IO=$IO ST=$ST make run-ml-hls
        done
    done
done
#for i in 64; do
#    for o in 64; do
#        for r in $(seq 1 $i); do
#            IN=$i OU=$o RF=$r make run-ml-hls
#        done
#    done
#done
