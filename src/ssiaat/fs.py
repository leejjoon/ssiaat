"""
Custom fsspec filesystem implementations for SPHEREx.
"""
import re
from fsspec.implementations.http import HTTPFileSystem
from fsspec.registry import register_implementation

class WebfsdHTTPFileSystem(HTTPFileSystem):
    """
    A simple extension of HTTPFileSystem that provides directory listing
    by parsing HTML index pages (specifically tailored for webfsd).
    """
    protocol = "webfsd"

    @classmethod
    def _strip_protocol(cls, path):
        # Translate webfsd:// to http:// so the parent HTTPFileSystem works
        if path.startswith("webfsd://"):
            return "http://" + path[len("webfsd://"):]
        return super()._strip_protocol(path)

    async def _ls(self, path, detail=False, **kwargs):
        session = await self.set_session()
        # The path here might already be stripped or absolute
        async with session.get(path) as response:
            if response.status != 200:
                return []
            text = await response.text()
            
        # Extract links from HTML
        items = set()
        # Find all hrefs
        hrefs = re.findall(r'href=["\']?([^"\'>\s]+)["\']?', text)
        for href in hrefs:
            # We want the name relative to path
            name = href.rstrip('/').split('/')[-1]
            # Filter out parent dir, absolute paths, and queries
            if name and not name.startswith(('?', '/', 'http')):
                items.add(name)
        
        # Also catch patterns like l2b-v... even if they are not in hrefs
        extra_names = re.findall(r'(l2b-v[\w-]+)', text)
        for name in extra_names:
            items.add(name)
            
        path = path.rstrip('/')
        if detail:
            return [{"name": f"{path}/{i}", "type": "directory", "size": None} for i in items]
        else:
            return [f"{path}/{i}" for i in items]

def register_fs():
    """Register custom filesystems with fsspec."""
    register_implementation("webfsd", WebfsdHTTPFileSystem, clobber=True)
