# def fn1():
#     import streamlit as st
#     from find_veterinaries import find_nearby_veterinaries
#     from vet import fetch_current_location

#     # Streamlit Interface
#     st.title("Veterinary Finder")

#     # Option to Manually Input Location
#     st.write("Enter Latitude and Longitude or Fetch Current Location")

#     latitude = st.number_input("Latitude", format="%.6f")
#     longitude = st.number_input("Longitude", format="%.6f")

#     # Button to Fetch Current Location
#     if st.button("Fetch Current Location"):
#         latitude, longitude = fetch_current_location()
#         if latitude and longitude:
#             st.success(f"Location fetched: {latitude}, {longitude}")
#         else:
#             st.error("Failed to fetch location. Please check your internet connection or try manually.")

#     # Button to Find Nearest Veterinary
#     # print(latitude, longitude)
#     if st.button("Find Nearest Veterinary"):
#         # if latitude and longitude:
#         find_nearby_veterinaries(latitude, longitude)
#         # else:
#         #     st.error("Please provide valid latitude and longitude.")



#------------------------------------------------------------------------------


def fn1():
    import streamlit as st
    from find_veterinaries import find_nearby_veterinaries
    from vet import fetch_current_location

    # Streamlit Interface
    st.title("Veterinary Finder")

    # Option to Manually Input Location
    st.write("Enter Latitude and Longitude or Fetch Current Location")

    latitude = st.number_input("Latitude", format="%.6f")
    longitude = st.number_input("Longitude", format="%.6f")

    # Button to Fetch Current Location
    if st.button("Fetch Current Location"):
        latitude, longitude = fetch_current_location()
        if latitude and longitude:
            st.success(f"Location fetched: {latitude}, {longitude}")
        else:
            st.error("Failed to fetch location. Please check your internet connection or try manually.")
            
    print(latitude, longitude)
    # Button to Find Nearest Veterinary
    if st.button("Find Nearest Veterinary"):
        # if latitude and longitude:
        # print(latitude, longitude)
        find_nearby_veterinaries(latitude, longitude)
        # else:
        #     st.error("Please provide valid latitude and longitude.")

