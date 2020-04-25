#!/usr/bin/env python3

import sys
sys.path.insert(0, 'lib')

import click
import matplotlib.pyplot as plt
from pmsp.simulator import simulator


@click.group()
def cli():
    pass

@click.command('run', short_help='Run simulation.')
def run():
    sim = simulator(filename='etc/config-basic.cfg')
    sim.train()

cli.add_command(run)

if __name__ == '__main__':
    cli()
