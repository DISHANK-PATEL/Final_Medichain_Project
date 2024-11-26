import streamlit as st
from login import collection
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def welcome():
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        username = st.session_state.get('username', 'User')

        # Fetch user's animals from the database
        user_data = collection.find_one({"username": username})
        animals = user_data.get("animals", []) if user_data else []

        # Ensure 'animals' is defined before using it
        if not animals:
            st.write("No animals found for this user.")
            return  # If no animals, exit early

        # Create layout with three columns for extra spacing
        col1, spacer, col2 = st.columns([1, 0.2, 3])  # Adjust ratios for more spacing

        # Display the logo in the left column
        with col1:
            st.image("logo.jpg", width=150)

        # Display welcome message in the right column with gradient background
        with col2:
            st.markdown(
                f"""
                <div style="padding-left: 10px; background: linear-gradient(to right, #6a11cb, #2575fc); padding: 20px; border-radius: 15px; text-align: center;">
                    <h2 style="color: white; font-size: 32px;">Welcome, {username}!</h2>
                    <p style="color: white; font-size: 18px;">Explore and manage your pets' information below.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display a summary chart if there are animals
        st.subheader("Your Animal Types Distribution")
        animal_types = [animal['type'] for animal in animals]
        animal_type_counts = pd.Series(animal_types).value_counts()

        # Display a bar chart with a dark background for the entire plot area
        fig, ax = plt.subplots(figsize=(4, 2))

        # Set the background color of the figure and axes to dark gray
        fig.patch.set_facecolor('#2c2f38')  # Darker outer background
        ax.set_facecolor('#2c2f38')  # Darker inner background for the chart

        # Plotting the bar chart with dark blue bars
        animal_type_counts.plot(kind='barh', ax=ax, color='#3b5998')

        # Set the labels and title in white
        ax.set_xlabel("Number of Animals", fontsize=8, color='white')
        ax.set_ylabel("Animal Type", fontsize=8, color='white')
        ax.set_title("Distribution of Animal Types", fontsize=10, color='white')

        # Set the tick labels in white for visibility
        plt.xticks(fontsize=6, color='white')
        plt.yticks(fontsize=6, color='white')

        # Set the spines (borders) color to light gray for better visibility
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

        # Ensure x-axis ticks are integers
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        st.pyplot(fig)
        st.write("A quick glance at the types of animals you own.")

        # Display animal details in a card-like style
        st.markdown("---")
        st.subheader("Your Animals")

        for animal in animals:
            # Hover effect for animal cards
            animal_card_style = """
                <style>
                    .expanderHeader:hover {
                        background-color: #f1f1f1;
                        cursor: pointer;
                    }
                </style>
            """
            st.markdown(animal_card_style, unsafe_allow_html=True)

            # Animal card with expandable details
            with st.expander(f"❤ {animal['name']}  - {animal['type']} 🐾"):
                # Display basic details without image and health status
                st.write(f"*Type:* {animal['type']}")
                st.write(f"*Age:* {animal.get('age', 'Unknown')} years")

                # Display and download each report under 'Important Documents'
                if "important_documents" in animal and isinstance(animal["important_documents"], list):
                    st.write("*Important Documents:*")
                    for report in animal["important_documents"]:
                        if isinstance(report, dict) and "name" in report and "pdf_data" in report:
                            # Use columns to align download and delete options side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label=f"📄 {report['name']}",
                                    data=base64.b64decode(report["pdf_data"]),
                                    file_name=report["name"],
                                    mime="application/pdf",
                                    help="Click to download the document",
                                    key=f"download_{animal['name']}_{report['name']}"  # Unique key for each button
                                )
                            with col2:
                                if st.button(f"🗑 Delete {report['name']}", key=f"delete_{animal['name']}_{report['name']}", help="Click to delete the document"):
                                    # Delete the report from MongoDB
                                    collection.update_one(
                                        {"username": username, "animals.name": animal["name"]},
                                        {"$pull": {"animals.$.important_documents": {"name": report["name"]}}}
                                    )
                                    st.success(f"Document '{report['name']}' deleted!")

                # Option to upload a new document
                new_report = st.file_uploader(f"Upload a new document for {animal['name']}", type="pdf",
                                              key=f"upload_{animal['name']}")
                if new_report and st.button(f"Add Document to {animal['name']}", key=f"add_report_{animal['name']}"):
                    report_data = {
                        "name": new_report.name,
                        "pdf_data": base64.b64encode(new_report.read()).decode('utf-8')
                    }
                    # Append the new report to MongoDB
                    collection.update_one(
                        {"username": username, "animals.name": animal["name"]},
                        {"$push": {"animals.$.important_documents": report_data}}
                    )
                    st.success(f"Document '{new_report.name}' added to {animal['name']}!")

            # Divider between each animal card
            st.divider()

    else:
        st.error("Please log in to access the dashboard.")
