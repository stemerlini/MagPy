import requests
s='https://www-amdis.iaea.org/FLYCHK/ZBAR/zp0'
ext = '.zvd'
direc = './'
A=1
while(A<80):
    if(A<10):
        AA = '0'+str(A)
    else:
        AA = str(A)
    url = s + AA + ext
    path = direc + str(A) + ext
    r = requests.get(url, allow_redirects=False)
    with open(path, 'wb') as f:
        f.write(r.content)
    A += 1
    