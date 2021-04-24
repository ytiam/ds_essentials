import site
def get_site_path():
    l = site.getsitepackages()
    return l[0]
print(get_site_path())
