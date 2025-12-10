from OpenSSL import crypto
import os

def generate_self_signed_cert():
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().O = "COD Application"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    os.makedirs("ssl", exist_ok=True)
    
    with open("ssl/cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("ssl/key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("SSL certificates generated in ssl/ folder")

if __name__ == "__main__":
    generate_self_signed_cert()
