#IOTYPE_RANGE=("io_parallel" "io_stream")
#STRATEGY_RANGE=("Latency" "Resource")
IOTYPE_RANGE=("io_parallel")
STRATEGY_RANGE=("Latency")

INPUT_RANGE=$(seq 4 16)
HEIGHT_RANGE=$INPUT_RANGE
WIDTH_RANGE=$INPUT_RANGE
CHANNELS_RANGE=$(seq 2 12)
OUTPUTS_RANGE=$(seq 2 12)
for iotype in ${IOTYPE_RANGE[@]}; do
    for strategy in ${STRATEGY_RANGE[@]}; do
        for height in ${HEIGHT_RANGE[@]}; do
            for width in ${WIDTH_RANGE[@]}; do
                for channels in ${CHANNELS_RANGE[@]}; do
                    for outputs in ${OUTPUTS_RANGE[@]}; do
                        lo=1
                        #$(($input + 1))
                        hi=1
                        #$(($input * $output))
                        for reusefactor in $(seq $lo $hi); do
                            H=$height W=$width C=$channels O=$outputs RF=$reusefactor IO=$iotype ST=$strategy make run-ml-hls
                        done
                    done
                done
            done
        done
    done
done
