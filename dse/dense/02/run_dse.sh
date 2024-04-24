#IOTYPE_RANGE=("io_parallel" "io_stream")
IOTYPE_RANGE=("io_parallel")
STRATEGY_RANGE=("Latency" "Resource")

INPUT_RANGE=$(seq 4 4)
OUTPUT_RANGE=$(seq 4 4)
for iotype in ${IOTYPE_RANGE[@]}; do
    for strategy in ${STRATEGY_RANGE[@]}; do
        for input in ${INPUT_RANGE[@]}; do
            for output in ${OUTPUT_RANGE[@]}; do
                lo=$(($input + 1))
                hi=$(($input * $output))
                for reusefactor in $(seq $lo $hi); do
                    IN=$input OU=$output RF=$reusefactor IO=$iotype ST=$strategy make run-ml-hls
                done
            done
        done
    done
done

IOTYPE_RANGE=("io_parallel" "io_stream")
STRATEGY_RANGE=("Latency" "Resource")

INPUT_RANGE=$(seq 8 8)
OUTPUT_RANGE=$(seq 8 8)
for iotype in ${IOTYPE_RANGE[@]}; do
    for strategy in ${STRATEGY_RANGE[@]}; do
        for input in ${INPUT_RANGE[@]}; do
            for output in ${OUTPUT_RANGE[@]}; do
                lo=$(($input + $input))
                hi=$(($input * $output))
                for reusefactor in $(seq $lo $hi $input); do
                    IN=$input OU=$output RF=$reusefactor IO=$iotype ST=$strategy make run-ml-hls
                done
            done
        done
    done
done

INPUT_RANGE=$(seq 16 16)
OUTPUT_RANGE=$(seq 16 16)
for iotype in ${IOTYPE_RANGE[@]}; do
    for strategy in ${STRATEGY_RANGE[@]}; do
        for input in ${INPUT_RANGE[@]}; do
            for output in ${OUTPUT_RANGE[@]}; do
                lo=$(($input + $input))
                hi=$(($input * $output))
                for reusefactor in $(seq $lo $hi $input); do
                    IN=$input OU=$output RF=$reusefactor IO=$iotype ST=$strategy make run-ml-hls
                done
            done
        done
    done
done

IOTYPE_RANGE=("io_stream")
STRATEGY_RANGE=("Latency" "Resource")

INPUT_RANGE=$(seq 32 32)
OUTPUT_RANGE=$(seq 32 32)
for iotype in ${IOTYPE_RANGE[@]}; do
    for strategy in ${STRATEGY_RANGE[@]}; do
        for input in ${INPUT_RANGE[@]}; do
            for output in ${OUTPUT_RANGE[@]}; do
                lo=$(($input + $input))
                hi=$(($input * $output))
                for reusefactor in $(seq $lo $hi $input); do
                    IN=$input OU=$output RF=$reusefactor IO=$iotype ST=$strategy make run-ml-hls
                done
            done
        done
    done
done
