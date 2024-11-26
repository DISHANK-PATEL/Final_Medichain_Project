
# def fnd():
#     import streamlit as st
#     from find_veterinaries import find_nearby_veterinaries

#     st.title("Find Nearby Veterinary Services")

#     # Input fields for user to enter latitude and longitude
#     latitude = st.text_input("Enter Latitude", "28.7041")  # Default is for Delhi
#     longitude = st.text_input("Enter Longitude", "77.1025")
#     location = f"{latitude},{longitude}"

#     # Input field for search radius
#     radius = st.slider("Search Radius (in meters)", 1000, 20000, 5000)

#     # Button to trigger the search
#     if st.button("Find Veterinary Services"):
#         api_key = "YOUR_GOOGLE_API_KEY"
#         results = find_nearby_veterinaries(location, radius, api_key)

#         # Display results
#         if results:
#             st.write("### Nearby Veterinary Services:")
#             for vet in results:
#                 st.write(f"**Name**: {vet['name']}")
#                 st.write(f"**Address**: {vet['address']}")
#                 st.write(f"**Rating**: {vet['rating']}")
#                 st.write(f"**User Ratings**: {vet['user_ratings_total']}")
#                 st.write("---")
#         else:
#             st.write("No nearby veterinary services found.")




import requests
import streamlit as st

def fetch_current_location():
    try:
        response = requests.get("https://ipinfo.io/loc")
        if response.status_code == 200:
            lat, lon = map(float, response.text.strip().split(','))
            return lat, lon
        else:
            st.error("Unable to fetch location data.")
            return None, None
    except Exception as e:
        st.error(f"Error fetching location: {e}")
        return None, None
