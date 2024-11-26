import streamlit as st
from streamlit_option_menu import option_menu
import login
import requests
import folium
from folium.plugins import MarkerCluster
from dotenv import load_dotenv
from streamlit_folium import folium_static
import os
from geopy.geocoders import Nominatim

# Set up page configuration
st.set_page_config(
    page_title="Medi-Chain",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session states for login and signup
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'signup_selected' not in st.session_state:
    st.session_state['signup_selected'] = False

# Display login or signup forms
if not st.session_state['logged_in']:
    if st.session_state['signup_selected']:
        login.signup()
        if st.button("Already have an account? Login here"):
            st.session_state['signup_selected'] = False
    else:
        login.login()
        if st.button("Don't have an account? Sign up here"):
            st.session_state['signup_selected'] = True
else:
    # Sidebar navigation menu
    from streamlit_option_menu import option_menu

    with st.sidebar:
        selected = option_menu(
            menu_title='Medi-Chain',
            options=[
                'Welcome', 'Disease & Breed Detection', 'Petcare ChatBot', 'Search Via Image',
                'Team Details', 'Feedback', 'Find Veterinary', 'Add Animal', 'Binary Classification Model',
                'Document-Analyzer', 'Logout'
            ],
            icons=[
                'house-door-fill', 'search', 'chat-right-fill', 'file-earmark-break-fill', 'info', 'star',
                'plus-circle', 'box-arrow-right', 'app', 'file-earmark-text-fill'
            ],
            menu_icon="🐶",
            default_index=0
        )

    # Handle menu selection
    if selected == 'Welcome':
        import welcome
        welcome.welcome()

    elif selected == 'Disease & Breed Detection':
        import model
        model.model()

    elif selected == 'Petcare ChatBot':
        import chatbot
        chatbot.chatbot()

    elif selected == 'Search Via Image':
        import prescription
        prescription.run()

    elif selected=='Document-Analyzer':
        import document_analyzer
        document_analyzer.main()


    elif selected == 'Feedback':
        import feedback
        feedback.feedback()

    elif selected == 'Team Details':
        import team
        team.team_details()

    elif selected == 'Add Animal':
        import add_animal
        add_animal.add_animal()

    elif selected == 'Binary Classification Model':
        import classify
        classify.model()

    elif selected == 'Logout':
        login.logout()
        st.session_state['logged_in'] = False
        st.session_state['signup_selected'] = False
        login.login()  # Redirect to login page after logout

    elif selected == 'Find Veterinary':
        # Load environment variables
        load_dotenv()
        API_KEY = os.getenv("PLACES_KEY")

        # Function to fetch nearby veterinary services
        def find_nearby_veterinaries(lat, lon, radius=5000, keyword="veterinary"):
            url = (
                f"https://maps.gomaps.pro/maps/api/place/nearbysearch/json?location="
                f"{lat},{lon}&radius={radius}&types=food&name={keyword}&key={API_KEY}"
            )
            try:
                response = requests.get(url)
                return response.json().get("results", [])
            except Exception as e:
                st.error(f"Error retrieving data from GoMaps.pro API: {e}")
                return []

        # Function to display map with markers
        def display_map(lat, lon, nearby_places):
            map_obj = folium.Map(location=[lat, lon], zoom_start=13)
            folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color='blue')).add_to(map_obj)

            marker_cluster = MarkerCluster().add_to(map_obj)
            for place in nearby_places:
                name = place.get("name")
                address = place.get("vicinity")
                place_lat = place.get("geometry", {}).get("location", {}).get("lat")
                place_lon = place.get("geometry", {}).get("location", {}).get("lng")
                place_url = place.get("place_id")

                if place_lat and place_lon:
                    folium.Marker(
                        [place_lat, place_lon],
                        popup=(
                            f"<b>{name}</b><br>{address}<br>"
                            f"<a href='https://www.google.com/maps/place/?q=place_id:{place_url}' target='_blank'>"
                            "Open in Google Maps</a>"
                        ),
                        icon=folium.Icon(color='green')
                    ).add_to(marker_cluster)

            return map_obj

        # Manual location input functionality
        location = st.text_input("Enter a location (city or address):", "")
        if location:
            geolocator = Nominatim(user_agent="vet_locator")
            location_obj = geolocator.geocode(location)

            if location_obj:
                lat, lon = location_obj.latitude, location_obj.longitude
                st.write(f"Showing results for {location_obj.address} (Lat: {lat}, Lon: {lon})")

                nearby_places = find_nearby_veterinaries(lat, lon)
                if nearby_places:
                    st.write("Nearby Veterinary Services:")
                    for place in nearby_places:
                        name = place.get("name")
                        address = place.get("vicinity")
                        place_url = place.get("place_id")
                        st.write(
                            f"{name} - {address} - "
                            f"[Google Maps Link](https://www.google.com/maps/place/?q=place_id:{place_url})"
                        )

                    # Display map with results
                    folium_static(display_map(lat, lon, nearby_places))
                else:
                    st.warning("No veterinary services found nearby.")
            else:
                st.error("Location not found. Please check the entered location.")
        else:
            st.write("Please enter a location.")
