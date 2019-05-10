'''
Analyze beam map data and produce some plot
'''

import numpy as np
import sys,os
import time
import beam_mapper_lib as b
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--file', '-f', help='Name of the file containing the noise data', type=str, required = True)
    parser.add_argument('--csv', '-c', help='Name of the csv file containing the beam map data, This is mandatory only if data are not embedded. When correctly given will embed data in the noise file', type=str)
    args = parser.parse_args()

    # Check if the data are present in the noise file, if not embed them

    if not b.check_beam_embedded(args.file):
        if args.csv is None:
            err_msg = "There are no beam map data embedded within the noise file. Please, provvide the --csv argument pointing to the beam map data"
            b.print_error(err_msg)
            exit()
        else:
            b.embed_beam_data(
                csv_filename = args.csv,
                noise_filename = args.file,
                verbose = True
            )


    #b.build_map(args.file, 24, 1.5, front_end = 'A_RX2', verbose = True)
    b.plot_beam_map(args.file)
