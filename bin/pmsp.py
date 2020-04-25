#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

import click
import matplotlib.pyplot as plt
from code import simulator


@click.group()
def cli():
    pass

@click.command('run', short_help='Run simulation.')
def run():
    sim = simulator()
    sim.train()
    sim.test()

cli.add_command(run)

if __name__ == '__main__':
    cli()
