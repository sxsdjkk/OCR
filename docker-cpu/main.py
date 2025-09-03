#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host='0.0.0.0',
        port=8008,
        reload=True,
        log_level="info",
        access_log=True,
    )


