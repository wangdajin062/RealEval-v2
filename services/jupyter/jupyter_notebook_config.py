import os
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_root = True
c.ServerApp.open_browser = False
c.ServerApp.token = os.environ.get("JUPYTER_TOKEN", "realeval")
c.ServerApp.password = ''
c.ServerApp.notebook_dir = '/workspace'
c.ServerApp.allow_origin = os.environ.get("JUPYTER_ALLOW_ORIGIN", "")
