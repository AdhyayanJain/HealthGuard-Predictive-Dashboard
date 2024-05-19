import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth

class FirebaseClient:
    def __init__(self, service_account_key_path):
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred)
        
    def create_user(self, email, password):
        user = auth.create_user(
            email=email,
            password=password
        )
        return user.uid
    
    def get_user(self, uid):
        user = auth.get_user(uid)
        return user
    
    def delete_user(self, uid):
        auth.delete_user(uid)
    