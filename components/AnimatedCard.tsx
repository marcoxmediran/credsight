import { useScrollAnimation } from '@/hooks/useScrollAnimation'

interface AnimatedCardProps {
  children: React.ReactNode
  delay?: number
  className?: string
}

export function AnimatedCard({ children, delay = 0, className = '' }: AnimatedCardProps) {
  const { isVisible, elementRef } = useScrollAnimation({ delay })

  return (
    <div
      ref={elementRef}
      className={`transition-all duration-[1200ms] ease-out ${
        isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
      } ${className}`}
    >
      {children}
    </div>
  )
}