
import csv
import collections
import json
import sys

import argparse


def print_freq_cnt(freq, str):
    print(str)
    for name,count in freq.items():
        print(name, count)


def merge_call_counts(rocprof_calls_freq, jaxprof_calls_freq, merged_csv_file):

    with open(merged_csv_file, 'w', newline='') as fd:
        csv_writer = csv.writer(fd)
        header = ["Name", "rocprof-calls", "jax-xla-calls"]
        csv_writer.writerow(header)

        for name, count in rocprof_calls_freq.items():
            if name in jaxprof_calls_freq:
                csv_writer.writerow((name, count, jaxprof_calls_freq[name]))
            else:
                csv_writer.writerow((name, count, -1))



def parse_jaxprof_json_dump(json_file_path, target_pid=701):
    try:
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)

            # Initialize a dictionary to count the frequency of each "name"
            freq = collections.defaultdict(int)

            # Process each event in the traceEvents list
            for event in json_data["traceEvents"]:

                # Check if the event not for host pid
                if event.get("pid") != target_pid:
                    # Increment the count for the name
                    kernel_name = event.get("name")
                    if kernel_name:
                        freq[kernel_name] += 1

            print_freq_cnt(freq, "\n kernal call frequency in jax-profiler : \n")

            return freq

    except FileNotFoundError:
        print(f"File '{json_file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_rocprof_kernel_trace(rocprof_kernel_trace, jaxprof_calls_freq):

    with open(rocprof_kernel_trace, 'r') as fd:
        csv_reader = csv.reader(fd)
        table = dict()
        freq = collections.defaultdict(int)
        header = next(csv_reader)
        print(header)
        line_idx = 1

        for idx, val in enumerate(header):
            table[val] = idx

        for line in csv_reader:
            line_idx += 1
            if len(line) > 0:
                try:
                    kernel_name = line[table["Kernel_Name"]]
                    dot_idx = kernel_name.rfind(".")
                    kernel_name=kernel_name[:dot_idx]
                    freq[kernel_name] += 1
                except Exception as err:
                    print(f"Line {line_idx}: {line}")
                    print(f"{err=}, {type(err)=}")

        print_freq_cnt(freq, "\n kernal call frequency in rocprofr : \n")

        dot_idx = rocprof_kernel_trace.rfind(".")
        merged_csv_file = rocprof_kernel_trace[:dot_idx] + ".stats.csv"
        merge_call_counts(freq, jaxprof_calls_freq, merged_csv_file)



def process_rocprof_hip_api_trace(rocprof_hip_trace, jaxprof_calls_freq):
     with open(rocprof_hip_trace, 'r') as fd:
        csv_reader = csv.reader(fd)
        table = dict()
        freq = collections.defaultdict(int)
        header = next(csv_reader)
        print(header)
        line_idx = 1

        for idx, val in enumerate(header):
            table[val] = idx

        for line in csv_reader:
            line_idx += 1
            if len(line) > 0:
                try:
                    func_name = line[table["Function"]]
                    freq[func_name] += 1
                except Exception as err:
                    print(f"Line {line_idx}: {line}")
                    print(f"{err=}, {type(err)=}")

        print_freq_cnt(freq, "\n hip-api call frequency in rocprofr : \n")
        dot_idx = rocprof_hip_trace.rfind(".")
        merged_csv_file = rocprof_hip_trace[:dot_idx] + ".stats.csv"
        merge_call_counts(freq, jaxprof_calls_freq, merged_csv_file)



def compile_rocprof_jaxprof_traces(jaxprof_trace, rocprof_kernel_trace, rocprof_hip_trace):

    jaxprof_calls_freq = parse_jaxprof_json_dump(jaxprof_trace)

    if rocprof_kernel_trace:
        process_rocprof_kernel_trace(rocprof_kernel_trace, jaxprof_calls_freq)
    if rocprof_hip_trace:
        process_rocprof_hip_api_trace(rocprof_hip_trace, jaxprof_calls_freq)



def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage with default values.')
    # Add arguments with default values
    parser.add_argument('--kernel_trace', type=str, default='', help='kernel-trace file)')
    parser.add_argument('--hip_trace', type=str, default='', help='hip-trace file)')
    parser.add_argument('--jaxprof_trace', type=str, default='', help='jaxprof-trace file)')

    # Parse arguments
    args = parser.parse_args()

    # Accessing and displaying the arguments
    print(f"--kernel_trace: {args.kernel_trace}")
    print(f"--hip_trace: {args.hip_trace}")
    print(f"--jaxprof_trace: {args.jaxprof_trace}")

    compile_rocprof_jaxprof_traces(args.jaxprof_trace, args.kernel_trace, args.hip_trace)


if __name__ == "__main__":
    main()
