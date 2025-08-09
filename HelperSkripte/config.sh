#!/bin/bash
# config.sh - Zentrale Konfiguration

# Basis Pfade
export MALMO_DIR="$HOME/Downloads/Malmo-0.37.0-Linux-Ubuntu-18.04-64bit_withBoost_Python3.6"
export MINECRAFT_DIR="$MALMO_DIR/Minecraft"
export PYTHON_DIR="$MALMO_DIR/Python_Examples"
export SCRIPTS_DIR="$PYTHON_DIR/HelperSkripte"

# Ports und Clients
export NUM_CLIENTS=2
export BASE_PORT=10000

# Farben
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export NC='\033[0m'