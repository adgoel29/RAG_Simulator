from loadfile import fileload

path=r"C:\Users\aditya\Desktop\pracsesh\sample.txt"

def chunking(path):
    content=fileload(path)
    chunk=content.split("\n\n")
    return chunk

if __name__=="__main__":
    path=r"C:\Users\aditya\Desktop\pracsesh\sample.txt"
    ans=chunking(path)
    print(ans)



