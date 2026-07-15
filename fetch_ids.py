import requests
import json

def get_transcripts(gene_symbol):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/symbol/mus_musculus/{gene_symbol}?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    
    if not r.ok:
        print(f"Error for {gene_symbol}: {r.status_code}")
        return []
        
    decoded = r.json()
    if not decoded:
        return []
    
    gene_id = decoded[0]['id']
    print(f"{gene_symbol} Gene ID: {gene_id}")
    
    ext2 = f"/lookup/id/{gene_id}?expand=1"
    r2 = requests.get(server+ext2, headers={ "Content-Type" : "application/json"})
    
    if not r2.ok:
        return []
        
    gene_data = r2.json()
    transcripts = [t['id'] for t in gene_data.get('Transcript', [])]
    return transcripts

print("Nanog:", get_transcripts("Nanog"))
print("Sox2:", get_transcripts("Sox2"))
