#!/usr/bin/python7
"""This script demonstrates the process for calling the download protocol.

What pray-tell is the download protocol? Well, it references a yaml file with
a bunch of urls and downloads the document contained therein and saves it to an
internal cache folder.

For Cardinal Health, this means something around 10 different pdf brochures
for what looks like different verticals within their business (guessing).

"""

# TODO - check module name
import ch_demo_2021

def main():
    protocol = ch_demo.datasets.pdfs.DownloadProtocol()
    protocol.download()

if __name__ == '__main__':
    main()
