"""Generate traffic for the city_map_2 network using centralized params."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.City_map.generate_traffic import generate_traffic
from src.params import NET_FILE, ROUTE_FILE, TRIPS_FILE


def main():
    generate_traffic(NET_FILE, TRIPS_FILE, ROUTE_FILE)


if __name__ == "__main__":
    main()
