#!/bin/bash
# launch_multi_minecraft.sh - Startet mehrere Minecraft Clients

# Lade Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo -e "${GREEN}🚀 Starte $NUM_CLIENTS Minecraft Clients...${NC}"

# Wechsle zum Minecraft Verzeichnis
cd "$MINECRAFT_DIR"

# Starte Clients
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    PORT=$((BASE_PORT + i))
    echo -e "${YELLOW}📦 Starte Client $i auf Port $PORT...${NC}"
    
    # NEU: Setze Java Options für weniger RAM
    export _JAVA_OPTIONS="-Xmx1G -Xms512M"  # Statt default 2G
    
    LOG_FILE="$MINECRAFT_DIR/client_${i}_port_${PORT}.log"
    PID_FILE="$MINECRAFT_DIR/client_${i}.pid"
    
    # Start mit nice für niedrigere Priorität
    nice -n 10 ./launchClient.sh -port $PORT > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo $PID > "$PID_FILE"
    echo -e "${GREEN}   ✓ Client $i läuft mit PID $PID (1GB RAM limit)${NC}"
    
    # WICHTIG: Mehr Zeit zwischen Starts!
    sleep 10  # Statt 3 Sekunden
done

echo -e "${GREEN}✅ Alle Clients gestartet!${NC}"
echo -e "${YELLOW}💡 Check Status mit: ./HelperSkripte/check_clients.sh${NC}"