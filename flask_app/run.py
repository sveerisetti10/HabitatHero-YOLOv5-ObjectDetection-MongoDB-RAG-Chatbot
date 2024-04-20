import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Here we use 8080 as default if not set
    port = int(os.environ.get('PORT', 8080))
    # Turn off debug mode when deploying to production
    app.run(host='0.0.0.0', port=port, debug=False)
