import { useState, useEffect, useRef } from 'react'

interface UseScrollAnimationOptions {
  delay?: number
  threshold?: number
}

export function useScrollAnimation({ delay = 0, threshold = 0.2 }: UseScrollAnimationOptions = {}) {
  const [isVisible, setIsVisible] = useState(false)
  const elementRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setTimeout(() => {
              setIsVisible(true)
            }, delay)
          } else {
            setIsVisible(false)
          }
        })
      },
      { threshold }
    )

    if (elementRef.current) {
      observer.observe(elementRef.current)
    }

    return () => {
      if (elementRef.current) {
        observer.unobserve(elementRef.current)
      }
    }
  }, [delay, threshold])

  return { isVisible, elementRef }
}