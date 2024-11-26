import streamlit as st
import webbrowser

def team_details():
    st.title("Our Team")

    team_members = [
        {
            "name": "DISHANK PATEL",
            "gmail": "dishankpatel16082003@gmail.com",
        },
        {
            "name": "HARSH SAHAY VERMA",
            "gmail": "9922103032@mail.jiit.ac.in",
        },
        {
            "name": "ADITYA KRISHNA",
            "gmail": "9922103045@mail.jiit.ac.in",
        },
        {
            "name": "NISHANT KUMAR",
            "gmail": "9922103053@mail.jiit.ac.in",
        }
    ]

    for member in team_members:
        st.header(member["name"])
        if st.button(f"Contact {member['name']} via Gmail"):
            webbrowser.open_new_tab(f"mailto:{member['gmail']}?subject=Regarding%20PawSome-AI%20App")

if __name__ == "__main__":
    team_details()
