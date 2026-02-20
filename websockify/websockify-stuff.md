# Notes to run Websockify

There are 2 main options to proxy with certificates: using NGINX or Apache2.

Must pay attention that the Websockify will install the Latest Numpy which may or not cause issues with Other softwares like IsaacSim.


This article seems to attempt doing in docker but commands differ from what the repository and docs recommend:

- [How to Run GUI Applications in Docker (X11 Forwarding and VNC)](https://oneuptime.com/blog/post/2026-01-16-docker-gui-apps-x11-vnc/view)

Old article from DigitalOcean exposes a single App:

- [How To Remotely Access GUI Applications Using Docker and Caddy on Ubuntu 20.04](https://www.digitalocean.com/community/tutorials/how-to-remotely-access-gui-applications-using-docker-and-caddy-on-ubuntu-20-04)


noVNC repo has this article:

- [Proxying with nginx](https://github.com/novnc/noVNC/wiki/Proxying-with-nginx)



## NGINX Reverse proxy

TODO: this needs review differs per cloud provider.

Install nginx and Apache2, review needed ???

```bash
sudo apt install -y nginx apache2-utils
```

Create /etc/nginx/sites-available/vnc with the following:

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        alias /usr/share/novnc/;
        index vnc.html;
    }

    location /websockify {
        proxy_http_version 1.1;
        proxy_pass http://127.0.0.1:6080;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 61s;
        proxy_buffering off;
    }
}
```

## Websockify

Install websockify, you can install websockify 



### Installing websockify Package Manager (APT) way

The apt way needs review as it might install a version that is not ideal or create issues with major dependencies in the machine.


This command needs review:

```bash
sudo apt install -y websockify
```

The apt-cache shows that python3-websockify and websockify are available

```bash
sudo apt-cache search websockify

python3-websockify - WebSockets support for any application/server - Python 3
websockify - WebSockets support for any application/server
```


### Installing websockify Python way

Download one of the releases or the latest development version, extract it and run ```python3 setup.py install``` as root in the directory where you extracted the files. Normally, this will also install **numpy** for better performance, if you don't have it installed already. However, numpy is optional. If you don't want to install numpy or if you can't compile it, you can edit setup.py and remove the install_requires=['numpy'], line before running ```python3 setup.py install```.

Afterwards, websockify should be available in your path. Run ```websockify --help``` to confirm it's installed correctly.

NOTE:

- Pay double attention to the **NUMPY** 

### Running websockify


Need to review the options within websockify

```bash
websockify --web /usr/share/novnc/ 6080 localhost:5901
```



## MISC



### Examples to Run

To show usage information:

```bash
./run --help
```

To listen on port 6080 (WebSocket) and forward to localhost port 5900 (TCP):

```bash
./run 6080 :5900
```

To enable the embedded static file webserver and forward to a remote server:

```bash
./run --web /path/to/novnc 6080 remote_hostname:5900
```


This example is based on:

- [Desktop Environment : VNC Client : noVNC](https://www.server-world.info/en/note?os=Ubuntu_22.04&p=desktop&f=8)

Install noVNC which is a VNC Client tool to connect to VNC server via Web Browser.


```bash
sudo apt -y install novnc python3-websockify python3-numpy
```

```bash
openssl req -x509 -nodes -newkey rsa:3072 -keyout novnc.pem -out novnc.pem -days 3650
```

```bash
websockify -D --web=/usr/share/novnc/ --cert=/home/ubuntu/novnc.pem 6080 localhost:5901
```



### websockify Package manager

Show websockify:

```bash
sudo apt show websockify
Package: websockify
Version: 0.10.0+dfsg1-2build1
Priority: optional
Section: universe/python
Origin: Ubuntu
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
Original-Maintainer: Debian OpenStack <team+openstack@tracker.debian.org>
Bugs: https://bugs.launchpad.net/ubuntu/+filebug
Installed-Size: 90.1 kB
Depends: python3-jwcrypto, python3-numpy, python3-websockify (>= 0.10.0+dfsg1-2build1), python3 (<< 3.11), python3 (>= 3.10~), libc6 (>= 2.34)
Homepage: https://pypi.python.org/pypi/websockify
Download-Size: 25.1 kB
APT-Sources: http://us.archive.ubuntu.com/ubuntu jammy/universe amd64 Packages
Description: WebSockets support for any application/server
 websockify was formerly named wsproxy and was part of the noVNC project.
 .
 At the most basic level, websockify just translates WebSockets traffic to
 normal socket traffic. Websockify accepts the WebSockets handshake, parses it,
 and then begins forwarding traffic between the client and the target in both
 directions.
 .
 Websockify supports all versions of the WebSockets protocol (Hixie and HyBi).
 The older Hixie versions of the protocol only support UTF-8 text payloads. In
 order to transport binary data over UTF-8 an encoding must used to encapsulate
 the data within UTF-8.
 .
 With Hixie clients, Websockify uses base64 to encode all traffic to and from
 the client. This does not affect the data between websockify and the server.
 .
 With HyBi clients, websockify negotiates whether to base64 encode traffic to
 and from the client via the subprotocol header (Sec-WebSocket-Protocol). The
 valid subprotocol values are 'binary' and 'base64' and if the client sends
 both then the server (the Python implementation) will prefer 'binary'. The
 'binary' subprotocol indicates that the data will be sent raw using binary
 WebSocket frames. Some HyBi clients (such as the Flash fallback and older
 Chrome and iOS versions) do not support binary data which is why the
 negotiation is necessary.
```

Show python3-websockify:

```bash
sudo apt show python3-websockify
Package: python3-websockify
Version: 0.10.0+dfsg1-2build1
Priority: optional
Section: python
Source: websockify
Origin: Ubuntu
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
Original-Maintainer: Debian OpenStack <team+openstack@tracker.debian.org>
Bugs: https://bugs.launchpad.net/ubuntu/+filebug
Installed-Size: 178 kB
Depends: python3-numpy, python3-pkg-resources, python3:any
Breaks: websockify (<< 0.8.0+dfsg1-14~)
Replaces: websockify (<< 0.8.0+dfsg1-14~)
Homepage: https://pypi.python.org/pypi/websockify
Download-Size: 41.9 kB
APT-Sources: http://us.archive.ubuntu.com/ubuntu jammy/main amd64 Packages
Description: WebSockets support for any application/server - Python 3
 websockify was formerly named wsproxy and was part of the noVNC project.
 .
 At the most basic level, websockify just translates WebSockets traffic to
 normal socket traffic. Websockify accepts the WebSockets handshake, parses it,
 and then begins forwarding traffic between the client and the target in both
 directions.
 .
 Websockify supports all versions of the WebSockets protocol (Hixie and HyBi).
 The older Hixie versions of the protocol only support UTF-8 text payloads. In
 order to transport binary data over UTF-8 an encoding must used to encapsulate
 the data within UTF-8.
 .
 With Hixie clients, Websockify uses base64 to encode all traffic to and from
 the client. This does not affect the data between websockify and the server.
 .
 With HyBi clients, websockify negotiates whether to base64 encode traffic to
 and from the client via the subprotocol header (Sec-WebSocket-Protocol). The
 valid subprotocol values are 'binary' and 'base64' and if the client sends
 both then the server (the Python implementation) will prefer 'binary'. The
 'binary' subprotocol indicates that the data will be sent raw using binary
 WebSocket frames. Some HyBi clients (such as the Flash fallback and older
 Chrome and iOS versions) do not support binary data which is why the
 negotiation is necessary.
 .
 This package provides the Python 3 module.
```



```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        alias /usr/share/novnc/;
        index vnc.html;
    }

    location /websockify {
        proxy_http_version 1.1;
        proxy_pass http://127.0.0.1:6080;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 61s;
        proxy_buffering off;
    }
}
```


```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        # Forward traffic to the local noVNC service 
        proxy_pass http://127.0.0.1:6080;

        # Required headers for websockify to maintain the connection
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Pass standard information about the client
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for smoother video rendering
        proxy_buffering off;
    }
}
```

Over port 443

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:6080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```