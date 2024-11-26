# MEDICHAIN 🐾

Medichain is a platform that combines Generative AI (GenAI), Machine Learning, and Blockchain technologies to help farmers efficiently manage their livestock. The platform leverages advanced AI to monitor animal health, diagnose illnesses, recommend treatments, and track overall well-being. Medichain aims to optimize farm management by providing data-driven insights, improving animal care, and reducing the risk of health issues. This innovative solution empowers farmers with actionable information for better decision-making, ultimately enhancing productivity and animal welfare in livestock management.

## Table of Contents  
- [Features](#features)  
- [Screenshots](#screenshots)  
- [Installation](#installation)  
- [Usage](#usage)  
- [License](#license)  

---

## Features  

### 1. **PetCare Multilingual Chatbot**  
- Supports multilingual input and response.  
- Search via **audio** input and receive **audio responses** related to pet care queries.  

### 2. **Disease & Breed Detection**  
- Utilizes the **Gemini API** for detecting diseases and breeds from images.  
- Upload an image of your pet to get precise insights.  

### 3. **Search Via Image**  
- Allows users to upload an image and ask anything related to it.  
- Uses advanced image recognition and query capabilities.  

### 4. **Find Veterinary**  
- Finds nearby veterinary stores and clinics.  
- Provides **Google Maps links** for easy navigation to the nearest veterinary services.  

### 5. **Add Animal**  
- Offers a dynamic dashboard to store and manage animal-related data.  
- Customize and organize information for each animal.  

### 6. **Binary Classification Model**  
- Assesses whether the condition of an animal is serious or not based on symptoms.  
- Helps prioritize care decisions.  

### 7. **Document Analyzer**  
- Upload multiple **PDF documents** related to animals (e.g., medical reports, vaccination records).  
- Extracts and interprets detailed information for better understanding.  

---


## Technologies Used  

### **Frontend**  
- **Framework:** Streamlit  

### **AI/ML Modules**  
- **Core Libraries:** TensorFlow, Scikit-learn, PyTorch  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Image Processing:** OpenCV, Pillow   
- **Optimization Algorithms:** SciPy 

### **Blockchain Integration**  
- **Smart Contracts:** Solidity  
- **Web3 Interaction:** Ether.js, MetaMask  

### **APIs**  
- OpenStreetMap API  
- Google Places API  
- Gemini API (Gemini Pro and Gemini 1.5 Flash) 

### **Database**  
- MongoDB  

### **Deployment**  
- Streamlit  

--- 

---

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-repo-name.git  
   cd your-repo-name  


## Screenshots
### Login/Signup Page
![image](https://github.com/user-attachments/assets/a0bb8ae2-df04-4997-ad88-cb3855d1fecd)

### Home Page
![image](https://github.com/user-attachments/assets/d6049d2a-68d6-4d35-bce3-740491c47f73)
![image](https://github.com/user-attachments/assets/eea1cfb8-f752-48a9-b5ad-8ad4bde5614c)



### Disease & Breed Detection
![image](https://github.com/user-attachments/assets/1c78773d-841e-4625-a271-6b5d9f3b0b85)
![image](https://github.com/user-attachments/assets/4cad468d-e583-4d90-840e-576229647b3a)



### Petcare ChatBot
![image](https://github.com/user-attachments/assets/d6b95183-f7bc-4d42-b0b4-f960959c0ec2)
![image](https://github.com/user-attachments/assets/1c67b2e3-93a4-4f7f-90f9-10908f4fa72f)



### Search Via Image
![image](https://github.com/user-attachments/assets/bd66f870-457d-4602-b50b-b6ac386e4521)

### Find Veterinary
![image](https://github.com/user-attachments/assets/c8476e23-7d8d-47cd-ba29-e735195f52ab)
![image](https://github.com/user-attachments/assets/7500d0c2-66ab-4a58-82a6-bc6ea56654cc)

### Document Analyzer
![image](https://github.com/user-attachments/assets/bb31cbbc-3d20-4326-b2ee-62a7d072d01d)


### Binary Classification Model
![image](https://github.com/user-attachments/assets/389ade86-cf6e-4092-a530-2b78fae2ce9b)
![image](https://github.com/user-attachments/assets/c76f640f-728a-46be-904f-d3f2e8094629)
![image](https://github.com/user-attachments/assets/6fcd6f54-f5d1-42e3-b5bb-b300eeab8653)
![image](https://github.com/user-attachments/assets/22c52927-5a05-4f74-ac62-d780b12fc080)

### Decentralised Storage on IPFS via Pinata

![ca4af4b2-07fa-4806-92d8-783068714dcf](https://github.com/user-attachments/assets/9e730a42-92e1-4e4f-bdfe-f4fe58a9ec4a)
![948e5db4-e6d3-4df9-b7ed-58da638de141](https://github.com/user-attachments/assets/ce4dfe70-f242-4b1b-8c44-42ba94998f69)
![809a3d2e-51f1-43fc-8f7c-4b76f9a8f35e](https://github.com/user-attachments/assets/c814229b-827a-4797-806a-8a66a3339291)
![d5e8a5cb-468b-412b-8d41-846a97900d49](https://github.com/user-attachments/assets/0e3b02ae-778e-42d5-870a-3d32ba961f9a)





### Figures

![c30f1011-919e-41bd-8c53-4aae85bdb714](https://github.com/user-attachments/assets/af2d77f3-8629-4c1b-8bf7-5e9b756c6054)
![08558c63-ffa5-43df-aebb-56ad20d3a1c4](https://github.com/user-attachments/assets/785d4200-ddd0-42e1-afd0-fb4c2abb77d7)
![4105c10c-b7f5-4289-8bbe-0777d08ec296](https://github.com/user-attachments/assets/76a85ebc-5f23-4ec7-8ace-d03a79752cc8)


## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash  
   git clone https://github.com/DISHANK-PATEL/Final_Medichain_Project.git  
    ```
2. Navigate into the project directory:
    ```bash
    cd minor_project-main
    ```
3. Install the required dependencies:
    ```bash
    python -m pip install -r requirements.txt
    ```

## Usage
Run the Streamlit application:
```bash
python -m streamlit run app.py
