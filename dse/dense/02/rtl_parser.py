import os
import csv
import re


def extract_rtl_info(base_dir, output_file):

    project_dir = os.path.join(base_dir, "hls4ml_prjs")

    if not os.path.isdir(project_dir):
        print(f"Directory not found: {project_dir}")
        return

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if os.stat(output_file).st_size == 0: 
            writer.writerow(["Latency", "Initiation Interval"])  

        for root, dirs, files in os.walk(project_dir):
            for directory in dirs:
                if not (directory.endswith(".gz") or directory.endswith(".yml")):
                    full_dir_path = os.path.join(root, directory)
                  
                    subfolder_path = os.path.join(full_dir_path, "dense_prj/dense.v1")

                    if os.path.isdir(subfolder_path):
                        
                        rtl_file = os.path.join(subfolder_path, "rtl.rpt")

                        if os.path.isfile(rtl_file):
                            with open(rtl_file, 'r') as f:
                                data_found = False
                                for line in f:
                                    if re.match(r"^  Design Total:", line, flags=re.IGNORECASE):
                                        data_found = True
                                        values = line.split()[3:5]  
                                        break

                                if data_found:
                                    writer.writerow(values)

script_dir = os.path.dirname(os.path.realpath(__file__))  

output_file = "rtl_info.csv"  

extract_rtl_info(script_dir, output_file)

