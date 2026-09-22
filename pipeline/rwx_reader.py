# Copied from the Lumenwood-World project (pipeline/rwx_reader.py, commit ebf0c72); only this header line was added. Used as an independent RWX checker.
"""Strict, read-only native RWX inspection for previews and geometry QA.

API: read_rwx(path, name=None) -> model dictionary, all coordinates in metres.
Unsupported commands fail with the source filename and line; source files are
never rewritten. Only the Python standard library is required.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, math, re, shlex, zipfile

class RWXError(ValueError):
    pass

def identity():return [[float(i==j) for j in range(4)] for i in range(4)]
def multiply(a,b):return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def transform(m,p):
    a=[*p,1];out=[sum(m[i][j]*a[j] for j in range(4)) for i in range(4)]
    if abs(out[3])<1e-12:raise RWXError('Transform maps a vertex to infinity')
    return [out[i]/out[3] for i in range(3)]
def sub(a,b):return [a[i]-b[i] for i in range(3)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def length(a):return math.sqrt(sum(v*v for v in a))
def bounds(vs):
    lo=[min(v[i] for v in vs) for i in range(3)];hi=[max(v[i] for v in vs) for i in range(3)]
    return {'min':lo,'max':hi,'size':[hi[i]-lo[i] for i in range(3)]}

def triangulate(points):
    """Ear clipping preserves concave polygon interiors and source winding."""
    if len(points)==3:return [[0,1,2]]
    normal=[0.,0.,0.]
    for p,q in zip(points,points[1:]+points[:1]):
        normal[0]+=(p[1]-q[1])*(p[2]+q[2]);normal[1]+=(p[2]-q[2])*(p[0]+q[0]);normal[2]+=(p[0]-q[0])*(p[1]+q[1])
    if length(normal)<1e-12:return []
    drop=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=drop]
    ps=[[p[i] for i in axes] for p in points]
    area=sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(ps,ps[1:]+ps[:1]));sign=1 if area>0 else -1
    def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def inside(p,a,b,c):return all(sign*x>=-1e-11 for x in [turn(a,b,p),turn(b,c,p),turn(c,a,p)])
    remaining=list(range(len(ps)));result=[]
    while len(remaining)>3:
        found=False
        for j,ib in enumerate(remaining):
            ia=remaining[j-1];ic=remaining[(j+1)%len(remaining)];a,b,c=ps[ia],ps[ib],ps[ic]
            t=sign*turn(a,b,c)
            if abs(t)<1e-12:
                # Exactly collinear/repeated vertices carry no polygon area.
                del remaining[j];found=True;break
            if t<0:continue
            # RWX exporters join hole loops with a repeated bridge edge. The
            # duplicate endpoint is the same geometric corner, not an interior
            # obstacle to clipping an ear on the other copy of that corner.
            if any(inside(ps[k],a,b,c) and all(math.hypot(ps[k][0]-v[0],ps[k][1]-v[1])>1e-10 for v in [a,b,c]) for k in remaining if k not in [ia,ib,ic]):continue
            result.append([ia,ib,ic]);del remaining[j];found=True;break
        if not found:raise RWXError('Cannot triangulate polygon (self-intersection or unsupported hole)')
    if len(remaining)==3:result.append(remaining)
    triangle_area=sum(abs(turn(ps[t[0]],ps[t[1]],ps[t[2]])) for t in result)
    if abs(triangle_area-abs(area))>1e-7*max(1,abs(area)):
        raise RWXError('Triangulation changed polygon area; input requires additional topology support')
    return result

def default_material():
    return {'color':[1.,1.,1.],'texture':None,'mask':None,'opacity':1.,'surface':[.69,0.,0.],
            'lightSampling':'facet','geometrySampling':'solid','textureModes':['lit','foreshorten'],
            'materialModes':[],'textureAddressMode':'wrap','collision':True,'axisAlignment':'none'}

@dataclass
class Frame:
    base:list=field(default_factory=identity)
    matrix:list=field(default_factory=identity)
    material:dict=field(default_factory=default_material)
    vertices:list=field(default_factory=list)
    uv:list=field(default_factory=list)
    prelights:list=field(default_factory=list)
    prelight:object=None
    clump_tag:int=0

class Reader:
    def __init__(self,source,name):
        self.source=source;self.name=name;self.prototypes={};self.warnings=[];self.commands={};self.parts=[]
        self.materials={};self.material_keys={};self.current_line=0;self.degenerate=0

    def fail(self,message):raise RWXError(f'{self.source}:{self.current_line}: {message}')
    def floats(self,args,n):
        if len(args)!=n:self.fail(f'Expected {n} numeric arguments, got {args!r}')
        try:values=[float(x) for x in args]
        except ValueError:self.fail(f'Invalid numeric arguments {args!r}')
        if not all(math.isfinite(x) for x in values):self.fail('Nonfinite numeric value')
        return values
    def warn(self,message):
        item=f'{self.source}:{self.current_line}: {message}'
        if item not in self.warnings:self.warnings.append(item)

    def material_id(self,mat,tag):
        mat=deepcopy(mat);mat['tag']=tag;mat['doubleSided']='double' in mat['materialModes']
        mat['unlit']='lit' not in mat['textureModes'] or (mat['surface'][1]==0 and mat['surface'][0]>=1)
        sig=json.dumps(mat,sort_keys=True,separators=(',',':'))
        if sig not in self.material_keys:
            key=f'{self.name}_mat{len(self.material_keys):03d}';self.material_keys[sig]=key;self.materials[key]=mat
        return self.material_keys[sig]

    def emit(self,frame,indices,tag,output):
        if frame.material['geometrySampling']!='solid':self.fail('Non-solid GeometrySampling cannot be emitted as a collision mesh')
        try:vs=[frame.vertices[i-1] for i in indices];uv=[frame.uv[i-1] for i in indices];prelight=[frame.prelights[i-1] for i in indices]
        except IndexError:self.fail(f'Face references vertex outside current clump: {indices}; count={len(frame.vertices)}')
        if any(i<=0 for i in indices):self.fail('RWX vertex indices are one-based positive integers')
        try:triangles=triangulate(vs)
        except RWXError as e:self.fail(str(e))
        good=[]
        for tri in triangles:
            a,b,c=[vs[i] for i in tri]
            if length(cross(sub(b,a),sub(c,a)))<1e-12:self.degenerate+=1
            else:good.append(tri)
        if not good:self.degenerate+=1;return
        mat=self.material_id(frame.material,tag)
        output.append({'material':mat,'vertices':vs,'triangles':good,'uv':uv,'prelight':prelight,
                       'collision':frame.material['collision'],'tag':tag,'clumpTag':frame.clump_tag,'sourceLine':self.current_line})

    def run(self,lines,frame,output,depth=0):
        if depth>64:self.fail('Excessive clump/prototype nesting')
        i=0;transforms=[];joints=[]
        while i<len(lines):
            number,tokens=lines[i];i+=1;self.current_line=number
            cmd=tokens[0].lower();args=tokens[1:];self.commands[cmd]=self.commands.get(cmd,0)+1
            if cmd in ['clumpbegin','protobegin']:
                if cmd=='clumpbegin' and args:self.fail('ClumpBegin takes no arguments')
                if cmd=='protobegin' and len(args)!=1:self.fail('ProtoBegin requires a name')
                ending='clumpend' if cmd=='clumpbegin' else 'protoend';level=1;j=i
                while j<len(lines):
                    c=lines[j][1][0].lower()
                    if c==cmd:level+=1
                    if c==ending:level-=1
                    if level==0:break
                    j+=1
                if j==len(lines):self.fail(f'Unclosed {cmd}')
                child=Frame(base=multiply(frame.base,frame.matrix) if cmd=='clumpbegin' else identity(),material=deepcopy(frame.material),prelight=frame.prelight)
                if cmd=='clumpbegin':self.run(lines[i:j],child,output,depth+1)
                else:
                    key=args[0].lower()
                    if key in self.prototypes:self.fail(f'Duplicate prototype {key}')
                    proto=[];self.run(lines[i:j],child,proto,depth+1);self.prototypes[key]=proto
                i=j+1;continue
            if cmd=='protoinstance':
                if len(args)!=1 or args[0].lower() not in self.prototypes:self.fail(f'Unknown or forward prototype {args!r}')
                world=multiply(frame.base,frame.matrix)
                for original in self.prototypes[args[0].lower()]:
                    part=deepcopy(original);part['vertices']=[transform(world,p) for p in original['vertices']];output.append(part)
                continue
            if cmd=='modelbegin' or cmd=='modelend':
                if args:self.fail(f'{cmd} takes no arguments')
            elif cmd=='transformbegin':transforms.append(deepcopy(frame.matrix))
            elif cmd=='transformend':
                if not transforms:self.fail('Unmatched TransformEnd')
                frame.matrix=transforms.pop()
            elif cmd=='jointtransformbegin':joints.append(True)
            elif cmd=='jointtransformend':
                if not joints:self.fail('Unmatched JointTransformEnd')
                joints.pop()
            elif cmd=='identityjoint':
                if args:self.fail('IdentityJoint takes no arguments')
                # Identity skeleton transform has no effect on a static mesh.
            elif cmd=='identity':frame.matrix=identity()
            elif cmd=='transform':
                values=self.floats(args,16)
                if values[15]==0:values[15]=1.;self.warn('Native-compatible Transform homogeneous final 0 interpreted as 1')
                if any(abs(values[n])>1e-12 for n in [3,7,11]):self.fail('Perspective transform is unsupported; expected an affine RWX matrix')
                frame.matrix=[[values[j*4+k] for j in range(4)] for k in range(4)]
            elif cmd in ['translate','scale','rotate']:
                t=identity()
                if cmd=='translate':
                    for k,v in enumerate(self.floats(args,3)):t[k][3]=v
                elif cmd=='scale':
                    for k,v in enumerate(self.floats(args,3)):t[k][k]=v
                else:
                    x,y,z,degrees=self.floats(args,4);norm=math.sqrt(x*x+y*y+z*z)
                    if norm<1e-12:self.fail('Rotate has a zero axis')
                    x/=norm;y/=norm;z/=norm;a=math.radians(degrees);c=math.cos(a);s=math.sin(a);q=1-c
                    t=[[c+x*x*q,x*y*q-z*s,x*z*q+y*s,0],[y*x*q+z*s,c+y*y*q,y*z*q-x*s,0],[z*x*q-y*s,z*y*q+x*s,c+z*z*q,0],[0,0,0,1]]
                frame.matrix=multiply(frame.matrix,t)
            elif cmd in ['vertex','vertexext']:
                if len(args)<3:self.fail('Vertex requires xyz')
                p=self.floats(args[:3],3);uv=[0.,0.];extra=args[3:];j=0;pre=frame.prelight
                while j<len(extra):
                    key=extra[j].lower();j+=1
                    if key=='uv':uv=self.floats(extra[j:j+2],2);j+=2
                    elif key=='prelight':pre=self.floats(extra[j:j+3],3);frame.prelight=pre;j+=3
                    else:self.fail(f'Unsupported {cmd} attribute {key!r}')
                frame.vertices.append(transform(multiply(frame.base,frame.matrix),p));frame.uv.append(uv);frame.prelights.append(deepcopy(pre))
            elif cmd in ['triangle','quad','polygon','triangleext','quadext','polygonext']:
                base=cmd.removesuffix('ext');tag=0;ids=args
                lower=[a.lower() for a in args]
                if 'tag' in lower:
                    ti=lower.index('tag')
                    if ti!=len(args)-2:self.fail('Tag must be a final integer face attribute')
                    try:tag=int(args[-1])
                    except ValueError:self.fail('Invalid face tag')
                    ids=args[:ti]
                try:ids=[int(x) for x in ids]
                except ValueError:self.fail('Noninteger face index')
                count=3 if base=='triangle' else 4
                if base=='polygon':
                    if not ids:self.fail('Polygon requires a count')
                    count=ids[0];ids=ids[1:]
                if count<3 or len(ids)!=count:self.fail(f'{cmd} vertex count mismatch')
                self.emit(frame,ids,tag,output)
            elif cmd=='color':frame.material['color']=self.floats(args,3)
            elif cmd=='surface':frame.material['surface']=self.floats(args,3)
            elif cmd in ['ambient','diffuse','specular']:frame.material['surface'][['ambient','diffuse','specular'].index(cmd)]=self.floats(args,1)[0]
            elif cmd=='opacity':frame.material['opacity']=self.floats(args,1)[0]
            elif cmd=='texture':
                if not args:self.fail('Texture requires a name or NULL')
                frame.material['texture']=None if args[0].lower()=='null' else args[0].lower()
                for k in ['mask','normal','specularMap']:frame.material[k]=None
                if (len(args)-1)%2:self.fail('Texture attributes require key/value pairs')
                for k in range(1,len(args),2):
                    key=args[k].lower()
                    if key not in ['mask','normal','specular']:self.fail(f'Unsupported Texture attribute {key}')
                    frame.material['specularMap' if key=='specular' else key]=args[k+1].lower()
            elif cmd in ['texturemode','texturemodes','addtexturemode','removetexturemode']:
                modes=[x.lower() for x in args]
                if any(x not in ['null','lit','foreshorten','filter'] for x in modes):self.fail(f'Unsupported texture mode {args}')
                if cmd.startswith('add'):frame.material['textureModes']=sorted(set(frame.material['textureModes']+modes)-{'null'})
                elif cmd.startswith('remove'):frame.material['textureModes']=[x for x in frame.material['textureModes'] if x not in modes]
                else:frame.material['textureModes']=sorted(set(modes)-{'null'})
            elif cmd in ['materialmode','materialmodes','addmaterialmode','removematerialmode']:
                modes=[x.lower() for x in args]
                if any(x not in ['null','double','none'] for x in modes):self.fail(f'Unsupported material mode {args}')
                if cmd.startswith('add'):frame.material['materialModes']=sorted(set(frame.material['materialModes']+modes)-{'null','none'})
                elif cmd.startswith('remove'):frame.material['materialModes']=[x for x in frame.material['materialModes'] if x not in modes]
                else:frame.material['materialModes']=sorted(set(modes)-{'null','none'})
            elif cmd=='collision':
                if len(args)!=1 or args[0].lower() not in ['on','off']:self.fail('Collision requires on/off')
                frame.material['collision']=args[0].lower()=='on'
            elif cmd in ['lightsampling','geometrysampling','textureaddressmode','axisalignment']:
                key={'lightsampling':'lightSampling','geometrysampling':'geometrySampling','textureaddressmode':'textureAddressMode','axisalignment':'axisAlignment'}[cmd]
                if len(args)!=1:self.fail(f'{cmd} requires one value')
                value=args[0].lower();valid={'lightsampling':['facet','vertex'],'geometrysampling':['solid','wireframe','pointcloud'],'textureaddressmode':['wrap','clamp','mirror'],'axisalignment':['none','zorientx','zorienty','xyz']}
                if value not in valid[cmd]:self.fail(f'Unsupported {cmd} {value}')
                frame.material[key]=value
                if cmd=='axisalignment' and value!='none':self.warn(f'Camera-facing {value} retained as material metadata; emitted mesh is its native rest orientation, renderer must apply billboarding')
            elif cmd=='tag':
                if len(args)!=1:self.fail('Clump Tag requires one integer')
                try:frame.clump_tag=int(args[0])
                except ValueError:self.fail('Invalid clump tag')
            elif cmd in ['hints','addhint','removehint']:
                self.warn(f'Preserved render hint without topology changes: {cmd} {" ".join(args)}')
                frame.material.setdefault('hints',[]).append([cmd,*args])
            else:self.fail(f'Unsupported command {tokens[0]!r}; geometry was not silently omitted')
        if transforms:self.fail('Unclosed TransformBegin')
        if joints:self.fail('Unclosed JointTransformBegin')

    def parse(self,text):
        lines=[]
        for n,line in enumerate(text.splitlines(),1):
            stripped=line.lstrip()
            # AW metadata extensions prefixed #! are executable, unlike comments.
            if stripped.startswith('#!'):line=stripped[2:]
            lexer=shlex.shlex(line,posix=True);lexer.whitespace_split=True;lexer.commenters='#';lexer.escape=''
            try:tokens=list(lexer)
            except ValueError as e:self.current_line=n;self.fail(str(e))
            if tokens:lines.append((n,tokens))
        if not lines or lines[0][1][0].lower()!='modelbegin' or lines[-1][1][0].lower()!='modelend':self.fail('Expected one ModelBegin/ModelEnd envelope')
        for envelope in ['modelbegin','modelend']:
            if sum(tokens[0].lower()==envelope for _,tokens in lines)!=1:self.fail(f'Expected exactly one {envelope}')
        self.run(lines,Frame(),self.parts)
        if not self.parts:self.fail('No renderable geometry')
        # Merge compatible face batches while retaining exact per-vertex UV/color data.
        grouped={}
        for p in self.parts:
            key=(p['material'],p['collision'],p['tag'],p['clumpTag'])
            dst=grouped.setdefault(key,{'material':p['material'],'vertices':[],'triangles':[],'uv':[],'prelight':[],'collision':p['collision'],'tag':p['tag'],'clumpTag':p['clumpTag']})
            off=len(dst['vertices']);dst['vertices'].extend([[v*10 for v in xyz] for xyz in p['vertices']]);dst['uv'].extend(p['uv']);dst['prelight'].extend(p['prelight']);dst['triangles'].extend([[i+off for i in t] for t in p['triangles']])
        parts=list(grouped.values());allvs=[v for p in parts for v in p['vertices']]
        inspection={'floorTriangles':[],'signFaces':[]}
        for pi,p in enumerate(parts):
            p['bounds']=bounds(p['vertices'])
            if not any(v is not None for v in p['prelight']):p.pop('prelight')
            for ti,t in enumerate(p['triangles']):
                vs=[p['vertices'][i] for i in t];n=cross(sub(vs[1],vs[0]),sub(vs[2],vs[0]));l=length(n);n=[x/l for x in n]
                if p['collision'] and n[1]>.5:inspection['floorTriangles'].append({'part':pi,'triangle':ti,'normal':n,'bounds':bounds(vs)})
            if p['tag']==100:
                a,b,c=[p['vertices'][i] for i in p['triangles'][0]];n=cross(sub(b,a),sub(c,a));l=length(n)
                inspection['signFaces'].append({'part':pi,'normal':[x/l for x in n],'bounds':p['bounds'],'uvBounds':{'min':[min(v[i] for v in p['uv']) for i in range(2)],'max':[max(v[i] for v in p['uv']) for i in range(2)]}})
        if self.degenerate:self.warnings.append(f'{self.degenerate} zero-area faces/triangles have no renderable surface and were reported rather than emitted')
        used={p['material'] for p in parts}
        return {'name':self.name,'units':'metres','parts':parts,'materials':{k:v for k,v in self.materials.items() if k in used},'bounds':bounds(allvs),'inspection':inspection,'warnings':self.warnings,'commands':self.commands,'triangles':sum(len(p['triangles']) for p in parts),'prototypes':len(self.prototypes)}

def read_rwx(path,name=None):
    path=Path(path).resolve();data=path.read_bytes();member=None
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            members=[n for n in z.namelist() if n.lower().endswith('.rwx')]
            matches=[n for n in members if Path(n).stem.lower()==(name or path.stem).lower()]
            if len(matches)==1:member=matches[0]
            elif len(members)==1:member=members[0]
            else:raise RWXError(f'{path}: expected one unambiguous RWX member, got {members}')
            if z.getinfo(member).file_size>64*1024*1024:raise RWXError(f'{path}: RWX member exceeds64MiB inspection limit')
            data=z.read(member)
    try:text=data.decode('utf-8-sig')
    except UnicodeDecodeError:text=data.decode('cp1252')
    model=Reader(str(path)+(f'!{member}' if member else ''),name or path.stem.lower()).parse(text)
    model['source']={'path':str(path),'member':member,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rwxSha256':hashlib.sha256(data).hexdigest()}
    return model

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('paths',nargs='+',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    models=[];failures=[]
    for path in args.paths:
        try:models.append(read_rwx(path))
        except (RWXError,zipfile.BadZipFile,UnicodeError) as e:failures.append({'path':str(path),'error':str(e)})
    result=models[0] if len(models)==1 and not failures and len(args.paths)==1 else {'models':models,'failures':failures}
    output=json.dumps(result,separators=(',',':'),allow_nan=False)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output,encoding='utf-8')
    else:print(output)
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
