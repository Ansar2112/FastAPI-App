# FastAPI App

This is a app for learning purposes

### HTTP Status Codes

HTTP Status Codes - 3 Digit Numbers - Indicate result of a client's request

### Categories
- **2xx (Success):** The request was successfully received and processed  
- **3xx (Redirection):** Further action needs to be taken (e.g., redirect)  
- **4xx (Client Error):** Something is wrong with the request from the client  
- **5xx (Server Error):** Something went wrong on the server side  

---

### Common Status Codes

#### ✅ Success
- **200 OK**  
  - Standard success  
  - A `GET` or `POST` succeeded  

- **201 Created**  
  - Resource created  
  - After a `POST` that creates something  

- **204 No Content**  
  - Success, but no data returned  
  - After a `DELETE` request  

---

#### ⚠️ Client Errors
- **400 Bad Request**  
  - Malformed or invalid request  
  - Example: Missing field, wrong data type  

- **401 Unauthorized**  
  - No/invalid authentication  
  - Example: Login required  

- **403 Forbidden**  
  - Authenticated, but no permission  
  - Example: Logged in but not allowed  

- **404 Not Found**  
  - Resource doesn’t exist  
  - Example: Patient ID not in DB  
