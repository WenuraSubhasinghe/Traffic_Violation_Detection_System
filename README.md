# Traffic Violation Detection System – Backend

This is the backend service for the **Traffic Violation Detection System**, built with **FastAPI** and **MongoDB**.  
It provides APIs for accident detection, traffic data storage, and image/video analysis.

---

## 🚀 Features

- **FastAPI-based REST API** for high-performance backend
- **MongoDB** for storing traffic and detection data
- **Accident detection pipeline** with image and video analysis
- **Docker-based MongoDB** setup for easy deployment
- **Virtual environment** support for Python dependencies

---

## 📋 Requirements

- **Python 3.10+**
- **pip** (Python package manager)
- **Docker** (for MongoDB)
- **Git** (optional, for version control)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Traffic_Violation_Detection_System
```

### 2. Create and Activate Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up MongoDB with Docker

**Start MongoDB:**
```bash
docker compose up -d
```

- This starts MongoDB with a root username/password (`root/example`)
- Data is persisted in `mongo_data` volume
- If `docker compose` is not available, use `docker-compose`

**Access MongoDB shell:**
```bash
docker exec -it traffic_mongo mongosh -u root -p example --authenticationDatabase admin
```

### 5. Run the FastAPI Server

**Development Mode:**
```bash
uvicorn app.main:app --reload
```

Server will start at: `http://127.0.0.1:8000`

**Production Mode (Optional):**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📚 API Documentation

FastAPI provides interactive API documentation:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## 🛠️ Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **POST** | `/traffic` | Add traffic data |
| **POST** | `/accidents/run` | Run accident detection on a video |
| **POST** | `/accidents/test-image` | Run accident detection on a single image |
| **GET** | `/traffic` | Retrieve stored traffic data |

---

## 📁 Outputs

Annotated images/videos are saved in the `outputs/` directory and served via:
```
http://127.0.0.1:8000/static/<filename>
```

---

## 🔧 Additional Notes

### Adding New Dependencies

Install new packages in the active virtual environment and update requirements:

```bash
pip install <package>
pip freeze > requirements.txt
```

### Project Structure

- The `.gitignore` file excludes `venv/` and `outputs/` from version control
- MongoDB data persists in Docker volumes

### Stopping Services

To stop and remove MongoDB container:
```bash
docker compose down
```

---

## 🐛 Troubleshooting

- **Docker not running:** Ensure Docker is started before launching MongoDB
- **Port conflicts:** Verify ports 8000 (FastAPI) and 27017 (MongoDB) are available
- **Python version:** Confirm Python 3.10+ is installed
- **Dependencies:** Ensure all packages are installed in the virtual environment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---