#!/usr/bin/env python3
"""
Browser Automation Service using Playwright
Provides headless Chromium automation for web scraping, testing, and interaction.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from playwright.async_api import async_playwright, Browser, Page

app = FastAPI(title="Browser Automation Service", version="0.1.0")

browser: Optional[Browser] = None


class NavigateRequest(BaseModel):
    url: str
    wait_until: str = "networkidle"
    timeout: int = 30000


class ScreenshotRequest(BaseModel):
    url: Optional[str] = None
    full_page: bool = False
    format: str = "png"


class ExtractRequest(BaseModel):
    url: str
    selector: Optional[str] = None
    extract_text: bool = True
    extract_html: bool = False


@app.on_event("startup")
async def startup():
    global browser
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)


@app.on_event("shutdown")
async def shutdown():
    global browser
    if browser:
        await browser.close()


@app.get("/health")
async def health():
    return {"status": "ok", "browser": "chromium", "ready": browser is not None}


@app.get("/healthz")
async def healthz():
    """Health endpoint matching gateway expectations."""
    return {"status": "ok", "browser": "chromium", "ready": browser is not None}


@app.post("/navigate")
async def navigate(req: NavigateRequest):
    """Navigate to a URL and return page info."""
    if not browser:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        page = await browser.new_page()
        await page.goto(req.url, wait_until=req.wait_until, timeout=req.timeout)
        title = await page.title()
        url = page.url
        await page.close()
        return {"url": url, "title": title, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screenshot")
async def screenshot(req: ScreenshotRequest):
    """Take a screenshot of a page."""
    if not browser:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        page = await browser.new_page()
        if req.url:
            await page.goto(req.url)
        screenshot_bytes = await page.screenshot(full_page=req.full_page, type=req.format)
        await page.close()
        from fastapi.responses import Response
        return Response(content=screenshot_bytes, media_type=f"image/{req.format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract(req: ExtractRequest):
    """Extract content from a page."""
    if not browser:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        page = await browser.new_page()
        await page.goto(req.url)
        result: Dict[str, Any] = {}
        
        if req.extract_text:
            if req.selector:
                element = await page.query_selector(req.selector)
                result["text"] = await element.text_content() if element else None
            else:
                result["text"] = await page.inner_text("body")
        
        if req.extract_html:
            if req.selector:
                element = await page.query_selector(req.selector)
                result["html"] = await element.inner_html() if element else None
            else:
                result["html"] = await page.content()
        
        await page.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)

