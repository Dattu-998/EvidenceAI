from fastapi import Request, HTTPException

def get_current_user_id(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header. Use 'Bearer <user_id>' for this test environment.")

    token = auth_header.split(" ")[1]
    return token
