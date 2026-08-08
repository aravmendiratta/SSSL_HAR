import './globals.css'

export const metadata = {
  title: 'SSSL-HAR Research Studio',
  description: 'Recreating the IJCB 2025 Paper: Bridging Virtual Kinematic Synthesis & Multi-View Contrastive Learning (MVCL) for Flexible IMU Activity Recognition.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
