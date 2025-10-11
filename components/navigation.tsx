"use client"

import { useState, useEffect } from "react"
import { Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
      setIsMobileMenuOpen(false)
    }
  }

  return (
    <nav
      className={`fixed left-0 right-0 top-0 z-50 transition-all duration-300 ${
        isScrolled ? "border-b border-border bg-background/80 backdrop-blur-lg" : "bg-transparent"
      }`}
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
        <div className="text-xl font-bold">CREDSIGHT</div>

        {/* Desktop Navigation */}
        <div className="hidden items-center gap-8 md:flex">
          <button
            onClick={() => scrollToSection("features")}
            className="text-sm text-foreground/80 transition-colors hover:text-foreground cursor-pointer"
          >
            Features
          </button>
          <button
            onClick={() => scrollToSection("how-it-works")}
            className="text-sm text-foreground/80 transition-colors hover:text-foreground cursor-pointer"
          >
            How It Works
          </button>
          <button
            onClick={() => scrollToSection("about")}
            className="text-sm text-foreground/80 transition-colors hover:text-foreground cursor-pointer"
          >
            About
          </button>
          <Link href="/analyze">
            <button className="header-premium-button px-4 py-2 text-sm font-medium">
              Get Started
            </button>
          </Link>
          <style jsx>{`
            .header-premium-button {
              background: transparent;
              color: #fff;
              letter-spacing: 0.5px;
              border: 1px solid rgba(255, 255, 255, 0.2);
              border-radius: 6px;
              position: relative;
              overflow: hidden;
              cursor: pointer;
              transition: all 0.3s ease;
            }
            
            .header-premium-button:hover {
              background: linear-gradient(to right, #C471ED, #12C2E9);
              border-color: transparent;
              transform: scale(1.02);
              box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
          `}</style>
        </div>

        {/* Mobile Menu Button */}
        <button className="md:hidden" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
          {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMobileMenuOpen && (
        <div className="border-t border-border bg-background/95 backdrop-blur-lg md:hidden">
          <div className="flex flex-col gap-4 px-4 py-6">
            <button
              onClick={() => scrollToSection("features")}
              className="text-left text-sm text-foreground/80 transition-colors hover:text-foreground cursor-pointer"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("how-it-works")}
              className="text-left text-sm text-foreground/80 transition-colors hover:text-foreground cursor-pointer"
            >
              How It Works
            </button>
            <button
              onClick={() => scrollToSection("about")}
              className="text-left text-sm text-foreground/80 transition-colors hover:text-foreground cursor-pointer"
            >
              About
            </button>
            <Link href="/analyze">
              <button className="mobile-premium-button w-full px-4 py-2 text-sm font-medium">
                Get Started
              </button>
            </Link>
            <style jsx>{`
              .mobile-premium-button {
                background: transparent;
                color: #fff;
                letter-spacing: 0.5px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                position: relative;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.3s ease;
              }
              
              .mobile-premium-button:hover {
                background: linear-gradient(to right, #C471ED, #12C2E9);
                border-color: transparent;
                transform: scale(1.02);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
              }
            `}</style>
          </div>
        </div>
      )}
    </nav>
  )
}
