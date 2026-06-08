from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class LibrarrClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        files: dict[str, Path] | None = None,
    ):
        url = self.base_url + path
        if params:
            url += '?' + urllib.parse.urlencode(params)
        req_headers = {'Accept': 'application/json'}
        if self.api_key:
            req_headers['X-Api-Key'] = self.api_key
        if headers:
            req_headers.update(headers)
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode('utf-8')
            req_headers['Content-Type'] = 'application/json'
        elif files:
            boundary = '----HermesBookBoundary'
            req_headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
            chunks: list[bytes] = []
            for field, file_path in files.items():
                chunks.append(f'--{boundary}\r\n'.encode('utf-8'))
                chunks.append(
                    f'Content-Disposition: form-data; name="{field}"; filename="{file_path.name}"\r\n'.encode('utf-8')
                )
                chunks.append(b'Content-Type: application/octet-stream\r\n\r\n')
                chunks.append(file_path.read_bytes())
                chunks.append(b'\r\n')
            chunks.append(f'--{boundary}--\r\n'.encode('utf-8'))
            data = b''.join(chunks)

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode('utf-8', 'replace')
                ctype = resp.headers.get_content_type()
                if 'json' in ctype or body.strip().startswith('{'):
                    return json.loads(body)
                return {'status': resp.status, 'body': body}
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace') if hasattr(e, 'read') else ''
            return {'error': e.reason, 'status': e.code, 'body': body}

    def add_to_wishlist(self, title: str, author: str = '', media_type: str = 'audiobook'):
        return self._request('POST', '/api/wishlist', json_body={'title': title, 'author': author, 'media_type': media_type})

    def create_request(
        self,
        title: str,
        author: str = '',
        cover_url: str = '',
        description: str = '',
        year: str = '',
        series_name: str = '',
        series_position: str = '',
    ):
        return self._request(
            'POST',
            '/api/requests',
            json_body={
                'title': title,
                'author': author,
                'book_type': 'audiobook',
                'cover_url': cover_url,
                'description': description,
                'year': year,
                'series_name': series_name,
                'series_position': series_position,
            },
        )

    def search(self, q: str):
        return self._request('GET', '/api/search', params={'q': q})

    def import_goodreads_csv(self, csv_path: str | Path):
        return self._request('POST', '/api/import/goodreads', files={'file': Path(csv_path)})

    def library(self):
        return self._request('GET', '/api/library')
