import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.join(path.dirname(fileURLToPath(import.meta.url)),'dist');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.jpg':'image/jpeg','.png':'image/png','.wasm':'application/wasm','.bin':'application/octet-stream'};
http.createServer((request,response)=>{
  const url=new URL(request.url,'http://127.0.0.1');
  const route=url.pathname==='/'?'/index.html':url.pathname==='/video'?'/video.html':url.pathname;
  const file=path.resolve(root,'.'+decodeURIComponent(route));
  if(!file.startsWith(root+path.sep)){response.writeHead(403).end();return;}
  fs.stat(file,(error,stat)=>{
    if(error||!stat.isFile()){response.writeHead(404).end('Not found');return;}
    response.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','Content-Length':stat.size});
    fs.createReadStream(file).pipe(response);
  });
}).listen(8766,'127.0.0.1',()=>console.log('Aerial Lens: http://127.0.0.1:8766'));
