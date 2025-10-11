import { useEffect, useRef } from 'react'

interface MetallicTextProps {
  children: React.ReactNode
  className?: string
}

export default function MetallicText({ children, className = '' }: MetallicTextProps) {
  const textRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const element = textRef.current
    if (!element) return

    let animationId: number
    let time = 0

    const animate = () => {
      time += 0.01
      
      // Liquid metal wave effect
      const wave1 = Math.sin(time * 2) * 0.5 + 0.5
      const wave2 = Math.sin(time * 3 + Math.PI / 3) * 0.5 + 0.5
      const wave3 = Math.sin(time * 1.5 + Math.PI / 2) * 0.5 + 0.5
      
      element.style.setProperty('--wave1', wave1.toString())
      element.style.setProperty('--wave2', wave2.toString())
      element.style.setProperty('--wave3', wave3.toString())
      element.style.setProperty('--time', time.toString())
      
      animationId = requestAnimationFrame(animate)
    }

    animate()
    return () => cancelAnimationFrame(animationId)
  }, [])

  return (
    <>
      <style jsx global>{`
        @keyframes liquidFlow {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        @keyframes liquidShimmer {
          0%, 100% { filter: brightness(1) contrast(1.2); }
          25% { filter: brightness(1.3) contrast(1.5); }
          50% { filter: brightness(0.9) contrast(1.1); }
          75% { filter: brightness(1.2) contrast(1.4); }
        }
        
        .liquid-metal {
          background: linear-gradient(
            45deg,
            #1a1a1a 0%,
            #4a4a4a calc(20% + var(--wave1, 0.5) * 10%),
            #ffffff calc(35% + var(--wave2, 0.5) * 15%),
            #c0c0c0 calc(50% + var(--wave3, 0.5) * 10%),
            #2a2a2a calc(65% + var(--wave1, 0.5) * 8%),
            #ffffff calc(80% + var(--wave2, 0.5) * 12%),
            #1a1a1a 100%
          );
          background-size: 300% 300%;
          background-clip: text;
          -webkit-background-clip: text;
          color: transparent;
          animation: liquidFlow 4s ease-in-out infinite, liquidShimmer 2s ease-in-out infinite;
          position: relative;
          text-shadow: none;
        }
        
        .liquid-metal::before {
          content: attr(data-text);
          position: absolute;
          top: 0;
          left: 0;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.8) calc(30% + var(--wave1, 0.5) * 20%),
            transparent calc(60% + var(--wave2, 0.5) * 20%),
            rgba(255, 255, 255, 0.6) calc(90% + var(--wave3, 0.5) * 10%),
            transparent 100%
          );
          background-clip: text;
          -webkit-background-clip: text;
          color: transparent;
          animation: liquidFlow 3s ease-in-out infinite reverse;
          z-index: 1;
        }
        
        .liquid-metal::after {
          content: '';
          position: absolute;
          top: -2px;
          left: -2px;
          right: -2px;
          bottom: -2px;
          background: radial-gradient(
            ellipse at calc(20% + var(--wave1, 0.5) * 60%) calc(30% + var(--wave2, 0.5) * 40%),
            rgba(255, 255, 255, 0.1) 0%,
            transparent 70%
          );
          border-radius: 8px;
          z-index: -1;
          animation: liquidFlow 5s ease-in-out infinite;
        }
      `}</style>
      <div
        ref={textRef}
        className={`liquid-metal ${className}`}
        data-text={typeof children === 'string' ? children : 'CREDSIGHT'}
      >
        {children}
      </div>
    </>
  )
}