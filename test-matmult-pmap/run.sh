#!/bin/bash

# Check if exactly two arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <size>"
    echo "example 1: $0 32"
    exit 1
fi

size=$1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE=[device_tracer_rocm=5, rocm_tracer=5]

outdir=$PWD/results_${size}

if [ -d "$outdir" ]; then
    # The directory exists, so remove it
    rm -rf "$outdir"
    echo "Directory '$outdir' has been removed."
else
    echo "Directory '$outdir' has been removed."
fi

echo $outdir
mkdir -p ${outdir}

rocprofv2 --kernel-trace --hip-api --hip-activity --plugin file -d ${outdir} python3 test_convolution_pmap.py $size  2>&1 | tee ${outdir}/console.log

# Use find to search for the file
perfetto_gz_file=$(find ${outdir} -type f -name perfetto_trace.json.gz)
rocprof_kernel_trace_file=$(find ${outdir} -type f -name 'results_*.csv')
rocprof_hip_trace_file=$(find ${outdir} -type f -name hip_api_trace*.csv)


echo "File perfetto: $perfetto_gz_file, rocprof-kerne-trace: $rocprof_kernel_trace_file, rocprof-hipapi-trace: $rocprof_hip_trace_file"

# Check if the files were found
if [ -e "$perfetto_gz_file" ] && [ -e "$rocprof_kernel_trace_file" ]; then
    # Unzip the file
    gzip -d "${perfetto_gz_file}" && echo "File unzipped successfully."
    perfetto_file=$(find ${outdir} -type f -name 'perfetto_trace.json')

    # Process the files with the Python script
    python3 ../bin/profiler_data_parser.py --jaxprof_trace "$perfetto_file" --kernel_trace "$rocprof_kernel_trace_file" --hip_trace "$rocprof_hip_trace_file"  2>&1 | tee ${outdir}/prof_tool_console.log

else

    if [ -e "$perfetto_gz_file" ]; then
        echo "File: $rocprof_kernel_trace_file' not found."
    else
         echo "File: '$perfetto_gz_file' not found."
    fi

fi

