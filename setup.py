"""Editable installation of the public checkout and required adapter packages."""
from setuptools import setup,find_namespace_packages
packages=find_namespace_packages(where='src')
mapping={name:'src/'+name.replace('.','/') for name in packages}
for prefix in ['newbench','orchestration','benchmark_interfaces']:
    names=[prefix]+[prefix+'.'+x for x in find_namespace_packages(where=prefix)]
    packages.extend(names);mapping.update({name:name.replace('.','/') for name in names})
packages.append('topology_stability');mapping['topology_stability']='benchmark_interfaces/topology_stability'
setup(packages=packages,package_dir=mapping)
