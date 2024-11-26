import requests
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from dotenv import load_dotenv
from streamlit_folium import folium_static
import os
from geopy.geocoders import Nominatim

load_dotenv()
API_KEY = os.getenv("PLACES_KEY")


# Function to find nearby veterinary services
def find_nearby_veterinaries(lat, lon, radius=5000, keyword="veterinary"):
    lat = str(round(float(lat), 4))
    lon = str(round(float(lon), 4))

    url = (
        f"https://maps.gomaps.pro/maps/api/place/nearbysearch/json?location="
        f"{lat},{lon}&radius={radius}&types=food&name={keyword}&key={API_KEY}"
    )

    try:
        response = requests.get(url)
        results = response.json().get("results", [])
        return results
    except Exception as e:
        st.error(f"Error retrieving data from GoMaps.pro API: {e}")
        return []


# Function to display the map and add user's location marker
def display_map(lat, lon, nearby_places):
    # Create a map centered at the user's location
    m = folium.Map(location=[lat, lon], zoom_start=13)

    # Add a marker for the user's location
    folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color='blue')).add_to(m)

    # Marker cluster for nearby places
    marker_cluster = MarkerCluster().add_to(m)

    for place in nearby_places:
        name = place.get("name")
        address = place.get("vicinity")
        place_lat = place.get("geometry", {}).get("location", {}).get("lat")
        place_lon = place.get("geometry", {}).get("location", {}).get("lng")
        place_url = place.get("place_id")

        if place_lat and place_lon:
            # Add each veterinary service as a marker on the map
            folium.Marker(
                [place_lat, place_lon],
                popup=f"<b>{name}</b><br>{address}<br><a href='https://www.google.com/maps/place/?q=place_id:{place_url}' target='_blank'>Open in Google Maps</a>",
                icon=folium.Icon(color='green')
            ).add_to(marker_cluster)

    return m


# Main function
def main():
    st.title("Find Nearby Veterinary Services")

    location_mode = st.radio("How would you like to provide your location?", ("Use My Current Location", "Enter Manually"))

    if location_mode == "Use My Current Location":
        st.write("Click the button below to allow access to your location.")
        html_code = """
        <script>
            function getLocation() {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;
                        const inputLat = document.getElementById("latitude");
                        const inputLon = document.getElementById("longitude");
                        inputLat.value = lat;
                        inputLon.value = lon;
                        document.getElementById("submit-location").click();
                    },
                    (error) => {
                        alert("Unable to fetch location. Please allow location access.");
                    }
                );
            }
        </script>
        <button onclick="getLocation()">Get My Location</button>
        <form>
            <input type="hidden" id="latitude" name="latitude" />
            <input type="hidden" id="longitude" name="longitude" />
            <button id="submit-location" style="display:none" type="submit">Submit</button>
        </form>
        """
        st.components.v1.html(html_code)

        # Extract latitude and longitude from query parameters
        lat = st.experimental_get_query_params().get("latitude", [None])[0]
        lon = st.experimental_get_query_params().get("longitude", [None])[0]

        if lat and lon:
            lat, lon = float(lat), float(lon)
            st.write(f"Using current location: Lat: {lat}, Lon: {lon}")

            # Fetch nearby veterinary services
            nearby_places = find_nearby_veterinaries(lat, lon)

            if nearby_places:
                st.write("Nearby Veterinary Services:")
                for place in nearby_places:
                    name = place.get("name")
                    address = place.get("vicinity")
                    place_url = place.get("place_id")
                    st.write(
                        f"{name} - {address} - [Google Maps Link](https://www.google.com/maps/place/?q=place_id:{place_url})")

                # Display map with user location and nearby places
                map_obj = display_map(lat, lon, nearby_places)
                folium_static(map_obj)

            else:
                st.warning("No veterinary services found nearby.")
        else:
            st.info("Waiting for location data...")

    elif location_mode == "Enter Manually":
        location = st.text_input("Enter a location (city or address):", "")

        if location:
            geolocator = Nominatim(user_agent="vet_locator")
            location_obj = geolocator.geocode(location)

            if location_obj:
                lat, lon = location_obj.latitude, location_obj.longitude
                st.write(f"Showing results for {location_obj.address} (Lat: {lat}, Lon: {lon})")

                # Fetch nearby veterinary services
                nearby_places = find_nearby_veterinaries(lat, lon)

                if nearby_places:
                    st.write("Nearby Veterinary Services:")
                    for place in nearby_places:
                        name = place.get("name")
                        address = place.get("vicinity")
                        place_url = place.get("place_id")
                        st.write(
                            f"{name} - {address} - [Google Maps Link](https://www.google.com/maps/place/?q=place_id:{place_url})")

                    # Display map with user location and nearby places
                    map_obj = display_map(lat, lon, nearby_places)
                    folium_static(map_obj)

                else:
                    st.warning("No veterinary services found nearby.")
            else:
                st.error("Location not found. Please check the entered location.")
        else:
            st.write("Please enter a location.")


if __name__ == "__main__":
    main()
