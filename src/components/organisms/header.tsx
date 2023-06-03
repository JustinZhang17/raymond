// External Imports
import { motion } from "framer-motion";

const Header = () => {
  return (
    <header className="flex items-center justify-between px-6 lg:px-48 mt-5 ">
      <div className="flex items-center justify-center">
        <img src="icon.png" className="h-16 py-2" alt="Raymond Logo" />
        <div className="hidden sm:block text-xl font-bold tracking-wider p-2 text-black md:text-4xl ml-3">
          Raymond
        </div>
      </div>
      <span className="hidden sm:flex">
        <motion.a
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          transition={{ duration: 0.1, ease: "easeOut" }}
          href="https://blog.justinjzhang.com/raymond"
          className="text-black hover:bg-gray-200 hover:text-sky-600 px-4 py-2 rounded-md text-lg font-bold flex mr-5"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-7 h-7 mr-2"
            stroke-width="1.5"
          >
            <path d="M11.25 4.533A9.707 9.707 0 006 3a9.735 9.735 0 00-3.25.555.75.75 0 00-.5.707v14.25a.75.75 0 001 .707A8.237 8.237 0 016 18.75c1.995 0 3.823.707 5.25 1.886V4.533zM12.75 20.636A8.214 8.214 0 0118 18.75c.966 0 1.89.166 2.75.47a.75.75 0 001-.708V4.262a.75.75 0 00-.5-.707A9.735 9.735 0 0018 3a9.707 9.707 0 00-5.25 1.533v16.103z" />
          </svg>
          <span className="hidden sm:block"> What is this?</span>
        </motion.a>
        <motion.a
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          transition={{ duration: 0.1, ease: "easeOut" }}
          href="https://github.com/JustinZhang17/raymond"
          className="text-black hover:bg-gray-200 hover:text-sky-600 px-4 py-2 rounded-md text-lg font-bold flex"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-7 h-7 mr-2"
            stroke-width="1.5"
          >
            <path
              fill-rule="evenodd"
              d="M2.25 6a3 3 0 013-3h13.5a3 3 0 013 3v12a3 3 0 01-3 3H5.25a3 3 0 01-3-3V6zm3.97.97a.75.75 0 011.06 0l2.25 2.25a.75.75 0 010 1.06l-2.25 2.25a.75.75 0 01-1.06-1.06l1.72-1.72-1.72-1.72a.75.75 0 010-1.06zm4.28 4.28a.75.75 0 000 1.5h3a.75.75 0 000-1.5h-3z"
              clip-rule="evenodd"
            />
          </svg>
          <span className="hidden sm:block">Source Code</span>
        </motion.a>
      </span>
    </header>
  );
};
export default Header;
